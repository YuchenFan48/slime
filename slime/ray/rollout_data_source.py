import copy
import os
import glob
import time
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer

# 假设这些类和函数是可用的
from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.types import Sample

# 假设 Sample, Dataset, load_function 已经定义或从 slime 导入

class RolloutDataSource:
    def __init__(self, args):
        self.args = args
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        
        # 1. 扫描文件
        self.data_files = []
        if os.path.isdir(args.prompt_data):
            self.data_files = glob.glob(os.path.join(args.prompt_data, "*.jsonl")) + \
                              glob.glob(os.path.join(args.prompt_data, "*.parquet"))
        else:
            self.data_files = [args.prompt_data]
        
        # 必须确保排序，保证多卡/恢复训练时顺序一致
        self.data_files = sorted([os.path.abspath(f) for f in self.data_files])
        
        # 打印调试信息
        if len(self.data_files) > 0:
            print(f"[DataSource] Found {len(self.data_files)} files.")
            print(f"[DataSource] First: {os.path.basename(self.data_files[0])}")
        
        self.total_files = len(self.data_files)
        self.current_file_index = 0 
        self.metadata = {}

        # 2. 初始化 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        if args.rollout_global_dataset and (d := args.dump_details) is not None:
            self.tokenizer.save_pretrained(Path(d) / "tokenizer")

        # === 3. 并行异步加载设置 ===
        self.dataset = None
        
        # 队列大小
        self.data_queue = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        
        # Worker 数量
        self.num_workers = 2 
        self.loader_thread = None

        # 【修改点 1】: 移除 __init__ 中的线程启动和文件加载，采用延迟启动策略
        # self._start_loader_thread(start_index=0)
        # self.load_next_file_from_queue()

        # 增加状态标志，标记数据源是否已启动（即线程是否已开始运行）
        self._has_started = False

    def _load_single_file(self, file_path, epoch_id):
        """
        单个文件的加载任务，将被提交给线程池
        """
        try:
            # 简化日志，只打印文件名
            fname = os.path.basename(file_path)
            # 为了调试目的，暂时保留写入日志的代码
            # 注意：在多线程环境中，这不是线程安全的写入方式
            with open('debug.log', 'a') as f:
                f.write(fname + '\n')
            # print(f"[AsyncLoader] Processing {fname}...") 
            
            # 导入 Dataset 必须在外部，这里只是示例调用
            ds = Dataset(
                file_path,
                tokenizer=self.tokenizer,
                max_length=self.args.rollout_max_prompt_len,
                prompt_key=self.args.input_key,
                label_key=self.args.label_key,
                metadata_key=self.args.metadata_key,
                tool_key=self.args.tool_key,
                seed=self.args.rollout_seed,
                shuffle=True
            )
            
            # 补 Shuffle
            if self.args.rollout_shuffle and not hasattr(ds, 'shuffled'):
                ds.shuffle(epoch_id)
            
            return ds
        except Exception as e:
            print(f"[AsyncLoader] Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _parallel_data_loader_daemon(self, start_index):
        """
        并行后台线程：
        维护一个 Future 字典，同时处理 N 个文件，但按顺序 Yield 结果。
        """
        submit_index = start_index 
        yield_index = start_index  
        
        futures = {} # index -> future
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            
            while not self.stop_event.is_set():
                # 1. 填满线程池
                while (len(futures) < self.num_workers) and (submit_index < self.total_files):
                    if self.stop_event.is_set(): break # Double check

                    file_path = self.data_files[submit_index]
                    # print(f"[AsyncLoader] Submitting {os.path.basename(file_path)}")
                    
                    futures[submit_index] = executor.submit(
                        self._load_single_file, file_path, self.epoch_id
                    )
                    submit_index += 1
                
                # 2. 按顺序获取结果并放入队列
                if yield_index in futures:
                    try:
                        # result() 会阻塞，直到该文件加载完成
                        ds = futures[yield_index].result()
                        del futures[yield_index]
                        
                        # 重要：在 put 之前再次检查 stop_event
                        if self.stop_event.is_set():
                            break

                        if ds is not None:
                            self.data_queue.put(ds) # 如果队列满了，会阻塞在这里
                        else:
                            # 加载失败的情况
                            self.data_queue.put(None) 
                            break
                        
                        yield_index += 1
                    except Exception as e:
                        print(f"[AsyncLoader] Exception in worker: {e}")
                        if yield_index in futures: del futures[yield_index]
                        yield_index += 1
                
                # 3. 检查是否全部完成
                elif submit_index >= self.total_files and len(futures) == 0:
                    self.data_queue.put(None) # 结束信号
                    break
                
                else:
                    # 暂时没有任务可提交，也没有结果可取
                    time.sleep(0.02)

    def _stop_background_thread(self):
        """
        安全停止后台线程
        """
        # 重置启动标志
        self._has_started = False
        
        if self.loader_thread is not None and self.loader_thread.is_alive():
            # 1. 发送停止信号
            self.stop_event.set()
            
            # 2. 必须消耗队列，防止线程卡在 queue.put() 上无法退出
            start_stop_time = time.time()
            while self.loader_thread.is_alive():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # 给线程一点时间退出
                self.loader_thread.join(timeout=0.1)
                
                if time.time() - start_stop_time > 5.0:
                    print("[DataSource] Warning: Background thread stuck, forcing continue.")
                    break

        # 3. 重置状态
        self.stop_event.clear()
        with self.data_queue.mutex:
            self.data_queue.queue.clear()

    def _start_loader_thread(self, start_index):
        """
        启动后台线程的统一入口，先停旧的，再开新的
        """
        self._stop_background_thread()
        
        if start_index >= len(self.data_files):
            print("[DataSource] Warning: start_index is beyond total files. Thread will start and immediately stop.")
            return

        print(f"[DataSource] Starting background loader from index {start_index} ({os.path.basename(self.data_files[start_index]) if start_index < len(self.data_files) else 'End'})")
        
        self.loader_thread = threading.Thread(
            target=self._parallel_data_loader_daemon, 
            args=(start_index,), 
            daemon=True
        )
        self.loader_thread.start()
        # 标记为已启动
        self._has_started = True


    def load_next_file_from_queue(self):
        """主线程从队列取货"""
        start_wait = time.time()
        
        next_ds = self.data_queue.get()
        
        wait_time = time.time() - start_wait
        if wait_time > 2.0:
            print(f"[DataSource] Waited {wait_time:.2f}s for next file.")
            
        self.dataset = next_ds
        if self.dataset is not None:
            # 这里的 current_file_index 是给 save() 用的，表示下一次如果崩溃了，存的是哪个
            self.current_file_index += 1
            print(f"[DataSource] Switched to dataset file index {self.current_file_index} (loaded)")
        else:
            print("[DataSource] Epoch finished (None received).")

    def get_samples(self, num_samples):
        
        # 【修改点 3】: 延迟启动检查 (Lazy Start)
        # 如果 load() 没被调用（或没找到文件），这里会执行，确保从 0 开始。
        if not self._has_started:
            print("[DataSource] Starting initial data load from index 0.")
            self._start_loader_thread(start_index=0)
            self.load_next_file_from_queue()
        
        prompt_samples = []
        remaining_samples = num_samples

        while remaining_samples > 0:
            if self.dataset is not None:
                current_len = len(self.dataset.samples)
                if self.sample_offset + remaining_samples <= current_len:
                    prompt_samples += self.dataset.samples[self.sample_offset: self.sample_offset + remaining_samples]
                    self.sample_offset += remaining_samples
                    remaining_samples = 0 
                else:
                    remaining_in_file = current_len - self.sample_offset
                    prompt_samples += self.dataset.samples[self.sample_offset:]
                    remaining_samples -= remaining_in_file
                    self.sample_offset = current_len 
                    self.load_next_file_from_queue()
                    if self.dataset is None: break 
                    self.sample_offset = 0
            
            if len(prompt_samples) < num_samples:
                if self.dataset is None: break
                continue

        # 如果跑完一轮了，进入下一个 Epoch
        if self.dataset is None:
            self.epoch_id += 1
            self.current_file_index = 0
            # 使用安全的启动方法重启
            self._start_loader_thread(start_index=0)
            self.load_next_file_from_queue()

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                # 假设 Sample 类可以被复制
                sample = copy.copy(prompt_sample) 
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
            
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset: return
        
        # 保存当前状态，current_file_index 指向的是“下一个还没完全处理完”或者“正在处理”的文件序号
        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
            "current_file_index": self.current_file_index, 
            "current_file_path": self.data_files[self.current_file_index - 1] if self.current_file_index > 0 else None,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset or self.args.load is None: return
        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path): return
        
        print(f"[DataSource] Loading state from {path}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        
        saved_file_index = state_dict.get("current_file_index", 0)
        
        # 恢复策略：回退一个文件，以确保数据不丢失
        resume_index = max(0, saved_file_index - 1) if saved_file_index > 0 else 0
        
        # 【修改点 2 核心】: 重启线程，直接从恢复点开始加载
        self._start_loader_thread(start_index=resume_index)
        
        # 立即读取，对齐状态 (加载的文件是 resume_index 指向的文件)
        self.load_next_file_from_queue()
        
        # 修正 current_file_index
        self.current_file_index = resume_index + 1
        
        # 此时 _has_started 已经被 _start_loader_thread 设置为 True
        
        print(f"[DataSource] Resumed. Current file index: {self.current_file_index}, Offset: {self.sample_offset}")


class RolloutDataSourceWithBuffer(RolloutDataSource):
    # 此类中的逻辑依赖于父类 RolloutDataSource 的行为，无需修改
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        # 假设 load_function 已经定义
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)
        if num_samples == 0: return samples
        # 父类的 get_samples 已经修改为懒启动模式
        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0: return []
        # 假设 pop_first 已经定义
        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        if not samples: return
        for i in range(len(samples)):
            self.buffer.append(samples[i])

    def get_buffer_length(self): return len(self.buffer)

# 假设 pop_first 已经定义
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples

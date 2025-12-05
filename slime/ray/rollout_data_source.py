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

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.types import Sample

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
            
        self.data_files = sorted([os.path.abspath(f) for f in self.data_files])
        self.total_files = len(self.data_files)
        self.current_file_index = 0 
        self.metadata = {}

        # 2. 初始化 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        if args.rollout_global_dataset and (d := args.dump_details) is not None:
            self.tokenizer.save_pretrained(Path(d) / "tokenizer")

        # === 3. 并行异步加载设置 ===
        self.dataset = None
        
        # 队列大小：控制“已处理完毕”的 Dataset 数量
        # 稍微加大一点，因为现在生产速度变快了
        self.data_queue = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        
        # 并行 Worker 数量：根据你的 CPU 核数和内存调整
        # 建议设为 2 到 4。太高可能会 OOM (内存溢出)，因为每个 Worker 都在内存里加载一个文件。
        self.num_workers = 2 
        
        # 启动后台守护线程
        self.loader_thread = threading.Thread(
            target=self._parallel_data_loader_daemon, 
            args=(0,), 
            daemon=True
        )
        self.loader_thread.start()

        # 加载第一个文件
        self.load_next_file_from_queue()

    def _load_single_file(self, file_path, epoch_id):
        """
        单个文件的加载任务，将被提交给线程池
        """
        try:
            # print(f"[AsyncLoader] Processing {os.path.basename(file_path)}...")
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
        submit_index = start_index # 下一个要提交给线程池的文件索引
        yield_index = start_index  # 下一个要放入队列的文件索引 (保证顺序)
        
        futures = {} # index -> future
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            
            while not self.stop_event.is_set():
                # 1. 填满线程池 (预取 N 个文件)
                # 只要线程池没满，且还有文件没提交，就继续提交
                while (len(futures) < self.num_workers) and (submit_index < self.total_files):
                    file_path = self.data_files[submit_index]
                    # 提交任务
                    futures[submit_index] = executor.submit(
                        self._load_single_file, file_path, self.epoch_id
                    )
                    submit_index += 1
                
                # 2. 按顺序获取结果并放入队列
                if yield_index in futures:
                    # 获取当前需要的文件 (如果还在处理，这里会阻塞等待它完成)
                    # 因为是并行处理，只要最慢的那个没卡住，这里通常会很快
                    ds = futures[yield_index].result()
                    del futures[yield_index]
                    
                    if ds is not None:
                        # 放入队列 (如果队列满了，会阻塞在这里，暂停提交新任务)
                        self.data_queue.put(ds)
                    else:
                        # 如果加载失败，传 None 终止或根据需求处理，这里为了稳健传 None
                        self.data_queue.put(None)
                        break
                    
                    yield_index += 1
                
                # 3. 检查是否全部完成
                elif submit_index >= self.total_files and len(futures) == 0:
                    self.data_queue.put(None) # 结束信号
                    break
                
                else:
                    # 暂时没有任务可提交，也没有结果可取 (极少见)，短暂休眠防空转
                    time.sleep(0.01)

    def load_next_file_from_queue(self):
        """主线程从队列取货"""
        # print("Waiting for next dataset...")
        start_wait = time.time()
        
        next_ds = self.data_queue.get()
        
        wait_time = time.time() - start_wait
        if wait_time > 1.0:
            print(f"[Warn] Main thread waited {wait_time:.2f}s for data. Consider increasing num_workers.")
            
        self.dataset = next_ds
        if self.dataset is not None:
            self.current_file_index += 1
        else:
            print("All files processed in this epoch.")

    def get_samples(self, num_samples):
        # 逻辑保持不变
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

        if self.dataset is None:
            self.epoch_id += 1
            self.current_file_index = 0
            self._restart_loader_thread()

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.copy(prompt_sample) 
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
            
        return samples
    
    def _restart_loader_thread(self):
        with self.data_queue.mutex:
            self.data_queue.queue.clear()
        
        # 重新启动，从 0 开始
        self.loader_thread = threading.Thread(
            target=self._parallel_data_loader_daemon, 
            args=(0,), 
            daemon=True
        )
        self.loader_thread.start()
        self.load_next_file_from_queue()

    # ... add_samples, save, load, RolloutDataSourceWithBuffer 保持不变 ...
    # 记得复制之前的 save/load/add_samples 代码
    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset: return
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
        
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        
        saved_file_index = state_dict.get("current_file_index", 0)
        resume_index = max(0, saved_file_index - 1) if saved_file_index > 0 else 0
        
        self.stop_event.set()
        with self.data_queue.mutex: self.data_queue.queue.clear()
        self.stop_event.clear()

        self.loader_thread = threading.Thread(
            target=self._parallel_data_loader_daemon, 
            args=(resume_index,), 
            daemon=True
        )
        self.loader_thread.start()
        self.load_next_file_from_queue()
        self.current_file_index = resume_index + 1

class RolloutDataSourceWithBuffer(RolloutDataSource):
    # ... 保持不变 ...
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)
        if num_samples == 0: return samples
        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0: return []
        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        if not samples: return
        for i in range(len(samples)):
            self.buffer.append(samples[i])

    def get_buffer_length(self): return len(self.buffer)

def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples

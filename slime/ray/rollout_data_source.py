import copy
import os
import glob
import time
import threading
import queue
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer

# 假设这些类和函数是可用的
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
        
        # === 0. 初始化日志文件 ===
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = f"debug_{timestamp_str}.log"
        
        # 写入起始分割线
        self._log(f"\n{'='*20} NEW RUN STARTED {'='*20}")
        
        # === 1. 扫描文件 ===
        self.data_files = []
        if os.path.isdir(args.prompt_data):
            self.data_files = glob.glob(os.path.join(args.prompt_data, "*.jsonl")) + \
                              glob.glob(os.path.join(args.prompt_data, "*.parquet"))
        else:
            self.data_files = [args.prompt_data]
        
        files_to_skip = {'train_data_b1016_p030.parquet', 'train_data_b0001_p031.parquet', "train_data_b0014_p005.parquet"} 
        
        # 记录过滤前的数量
        original_count = len(self.data_files)
        
        if len(self.data_files) < original_count:
            skipped_count = original_count - len(self.data_files)
            # 如果你有 _log 函数就用 _log，没有就用 print
            msg = f"[Init] Manually skipped {skipped_count} files found in blocklist: {files_to_skip}"
            if hasattr(self, '_log'):
                self._log(msg)
            else:
                print(msg, flush=True)

        # 执行过滤
        self.data_files = [
            f for f in self.data_files 
            if os.path.basename(f) not in files_to_skip
        ]

        # 按文件名倒序排序
        self.data_files = sorted(
            [os.path.abspath(f) for f in self.data_files], 
            key=lambda x: os.path.basename(x), 
        )
        
        self.total_files = len(self.data_files)
        self.current_file_index = 0 
        self.metadata = {}

        # 【DEBUG】打印文件列表检查
        self._log(f"[Init] Found {self.total_files} files.")
        if self.total_files > 0:
            self._log("[Init] Top 5 Files in sorted list:")
            for i, f in enumerate(self.data_files[:5]):
                self._log(f"    [{i}] {os.path.basename(f)}")
        else:
            self._log("[Init] WARNING: No files found!")

        # 2. 初始化 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        if args.rollout_global_dataset and (d := args.dump_details) is not None:
            path = Path(d) / "tokenizer"
            if not path.exists():
                self.tokenizer.save_pretrained(path)

        # === 3. 异步加载设置 ===
        self.dataset = None
        self.data_queue = queue.Queue(maxsize=3) 
        self.stop_event = threading.Event()
        self.num_workers = 3
        
        self.loader_thread = None
        self._has_started = False

    def _log(self, msg):
        """
        核心日志函数：同时打印到屏幕和写入文件
        """
        # 1. 打印到屏幕 (强制刷新)
        print(msg, flush=True)
        
        # 2. 追加到文件
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {msg}\n")
        except Exception as e:
            print(f"!!! Error writing to debug.log: {e}")

    def _load_single_file(self, file_path, epoch_id):
        fname = os.path.basename(file_path)
        # 【DEBUG】线程开始
        self._log(f"[Worker] START loading: {fname}")
        
        try:
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
            
            if self.args.rollout_shuffle and not hasattr(ds, 'shuffled'):
                ds.shuffle(epoch_id)
            
            # 【DEBUG】线程结束
            self._log(f"[Worker] DONE loading: {fname} | Samples: {len(ds.samples)}")
            return ds, fname
        except Exception as e:
            self._log(f"[Worker] ERROR loading {fname}: {e}")
            traceback.print_exc()
            return None, fname

    def _parallel_data_loader_daemon(self, start_index):
        submit_index = start_index 
        yield_index = start_index  
        futures = {} 
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            while not self.stop_event.is_set():
                # 提交任务
                while (len(futures) < self.num_workers) and (submit_index < self.total_files):
                    if self.stop_event.is_set(): break
                    file_path = self.data_files[submit_index]
                    futures[submit_index] = executor.submit(
                        self._load_single_file, file_path, self.epoch_id
                    )
                    submit_index += 1
                
                # 获取结果
                if yield_index in futures:
                    try:
                        ds, fname = futures[yield_index].result()
                        del futures[yield_index]
                        
                        if self.stop_event.is_set(): break

                        if ds is not None:
                            self.data_queue.put(ds) 
                        else:
                            self._log(f"[Daemon] Warning: File {fname} failed to load. Skipping.")
                        yield_index += 1
                    except Exception as e:
                        self._log(f"[Daemon] Exception in worker: {e}")
                        if yield_index in futures: del futures[yield_index]
                        yield_index += 1
                
                elif submit_index >= self.total_files and len(futures) == 0:
                    self.data_queue.put(None)
                    self._log("[Daemon] All files submitted and loaded.")
                    break
                else:
                    time.sleep(0.02)

    def _stop_background_thread(self):
        self._has_started = False
        if self.loader_thread is not None and self.loader_thread.is_alive():
            self.stop_event.set()
            start_stop_time = time.time()
            while self.loader_thread.is_alive():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass
                self.loader_thread.join(timeout=0.1)
                if time.time() - start_stop_time > 5.0:
                    self._log("[Stop] Thread stuck, forcing continue.")
                    break
        self.stop_event.clear()
        with self.data_queue.mutex:
            self.data_queue.queue.clear()

    def _start_loader_thread(self, start_index):
        self._stop_background_thread()
        if start_index >= len(self.data_files):
            return

        fname = os.path.basename(self.data_files[start_index])
        self._log(f"[Thread] Starting background loader from index {start_index} -> {fname}")
        
        self.loader_thread = threading.Thread(
            target=self._parallel_data_loader_daemon, 
            args=(start_index,), 
            daemon=True
        )
        self.loader_thread.start()
        self._has_started = True

    def load_next_file_from_queue(self):
        # 【DEBUG】主线程开始等待
        # self._log("[Main] Waiting for next dataset from queue...")
        next_ds = self.data_queue.get()
        
        self.dataset = next_ds
        if self.dataset is not None:
            self.current_file_index += 1
            if self.current_file_index <= len(self.data_files):
                fname = os.path.basename(self.data_files[self.current_file_index - 1])
                self._log(f"[Main] Switched to dataset: {fname} (Seq: {self.current_file_index}/{self.total_files})")
        else:
            self._log("[Main] Epoch finished (End of file list).")

    def get_samples(self, num_samples):
        if not self._has_started:
            self._log("[Main] First call, starting loader thread...")
            self._start_loader_thread(start_index=0)
            self.load_next_file_from_queue()
        
        prompt_samples = []
        remaining_samples = num_samples

        while remaining_samples > 0:
            if self.dataset is not None:
                current_len = len(self.dataset.samples)
                available_in_file = current_len - self.sample_offset
                
                if available_in_file >= remaining_samples:
                    # 当前文件够用
                    prompt_samples += self.dataset.samples[self.sample_offset: self.sample_offset + remaining_samples]
                    self.sample_offset += remaining_samples
                    remaining_samples = 0 
                else:
                    # 当前文件不够用，读完它
                    fname = "Unknown"
                    if self.current_file_index > 0:
                        fname = os.path.basename(self.data_files[self.current_file_index-1])
                    
                    self._log(f"[Main] FINISHED File: {fname}. Consumed {current_len} samples. Loading next...")
                    
                    prompt_samples += self.dataset.samples[self.sample_offset:]
                    remaining_samples -= available_in_file
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
            self._log(f"[Main] Epoch {self.epoch_id} starting, restarting file list.")
            self._start_loader_thread(start_index=0)
            self.load_next_file_from_queue()

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

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset: return
        
        current_fname = os.path.basename(self.data_files[self.current_file_index - 1]) if self.current_file_index > 0 else None
        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
            "current_file_name": current_fname,
            "current_file_index": self.current_file_index, 
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)
        # self._log(f"[Save] Saved state to {path}")

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset or self.args.load is None:
            return
        
        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            self._log(f"[Load] No checkpoint found at {path}, starting from scratch.")
            return
        
        self._log(f"[Load] Loading state from {path}")
        state_dict = torch.load(path, map_location="cpu")
        
        # 1. 基础状态恢复
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        
        saved_file_index = state_dict.get("current_file_index", 0)
        saved_file_name = state_dict.get("current_file_name", None)

        # 2. 定位恢复的起始文件索引 (resume_index)
        resume_index = -1
        
        # 优先通过文件名匹配（解决 579 变 551 的偏移问题）
        if saved_file_name:
            for i, f in enumerate(self.data_files):
                if os.path.basename(f) == saved_file_name:
                    resume_index = i
                    self._log(f"[Load] Found saved file '{saved_file_name}' at current index {resume_index}")
                    break
        
        # 如果文件名匹配失败，尝试使用旧的数字索引
        if resume_index == -1:
            if saved_file_index > 0:
                resume_index = max(0, saved_file_index - 1)
                self._log(f"[Load] WARNING: Filename match failed. Falling back to index {resume_index}")
            else:
                resume_index = 0
                self._log(f"[Load] Starting from index 0.")

        # 3. 彻底重置加载状态，准备从新位置启动
        # 先停止可能存在的后台线程
        self._stop_background_thread()
        
        # 设置主线程的 current_file_index。
        # 注意：load_next_file_from_queue() 内部会执行 += 1
        # 所以我们设为 resume_index，load 完之后它会变成正确的 resume_index + 1
        self.current_file_index = resume_index
        
        # 启动后台加载线程，从 resume_index 开始
        self._start_loader_thread(start_index=resume_index)
        
        # 从队列中取出对应的第一个数据集对象
        self.load_next_file_from_queue()
        
        # 4. 最终状态校验日志
        actual_fname = os.path.basename(self.data_files[self.current_file_index-1])
        self._log(f"[Load] Resumed successfully:")
        self._log(f"    - Epoch: {self.epoch_id}")
        self._log(f"    - File Seq: {self.current_file_index}/{self.total_files}")
        self._log(f"    - File Name: {actual_fname}")
        self._log(f"    - Offset: {self.sample_offset}")


class RolloutDataSourceWithBuffer(RolloutDataSource):
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

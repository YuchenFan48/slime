import copy
import os
import glob
import time
import threading
import queue
from pathlib import Path

import torch
from transformers import AutoTokenizer

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.types import Sample
from slime.utils.train_metric_utils import log_with_file

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
        # current_file_index 记录"主线程已经消费到第几个文件"
        self.current_file_index = 0 
        self.metadata = {}

        # 2. 初始化 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        if args.rollout_global_dataset and (d := args.dump_details) is not None:
            self.tokenizer.save_pretrained(Path(d) / "tokenizer")

        # === 3. 异步加载核心设置 ===
        self.dataset = None
        # 队列大小设为 2: 内存中最多保留 2 个 Dataset 对象 (处理好的)
        self.data_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        
        # 启动后台加载线程 (从第 0 个文件开始)
        self.loader_thread = threading.Thread(
            target=self._data_loader_daemon, 
            args=(0,), 
            daemon=True
        )
        self.loader_thread.start()

        # 主线程立即尝试获取第一个文件 (会阻塞直到第一个文件处理完)
        self.load_next_file_from_queue()

    def _data_loader_daemon(self, start_index):
        """
        后台守护线程：
        1. 读取文件路径
        2. 实例化 Dataset (此时会执行耗时的 Tokenize/Mask)
        3. 塞入 Queue 等待主线程取用
        """
        thread_file_index = start_index
        print(f"[AsyncLoader] Thread started from file index {thread_file_index}")
        
        while not self.stop_event.is_set():
            # 检查是否读完所有文件
            if thread_file_index >= self.total_files:
                self.data_queue.put(None) # 发送结束信号
                break 

            file_path = self.data_files[thread_file_index]
            try:
                # === 耗时操作在这里执行 (不占用 GPU 时间) ===
                ds = Dataset(
                    file_path,
                    tokenizer=self.tokenizer,
                    max_length=self.args.rollout_max_prompt_len,
                    prompt_key=self.args.input_key,
                    label_key=self.args.label_key,
                    metadata_key=self.args.metadata_key,
                    tool_key=self.args.tool_key,
                    seed=self.args.rollout_seed,
                    shuffle=self.args.rollout_shuffle, # Shuffle 也可以在这里做
                    # 其他参数...
                )
                
                # 如果 Dataset 内部没做 shuffle，这里补一个
                if self.args.rollout_shuffle and not hasattr(ds, 'shuffled'):
                    ds.shuffle(self.epoch_id)
                
                # 放入队列。如果队列满了(maxsize=2)，线程会在这里自动阻塞挂起
                self.data_queue.put(ds)
                
                thread_file_index += 1
                
            except Exception as e:
                print(f"[AsyncLoader] Error loading {file_path}: {e}")
                import traceback
                traceback.print_exc()
                # 发生错误也要发信号，防止主线程死锁
                self.data_queue.put(None)
                break
    
    def load_next_file_from_queue(self):
        """
        主线程动作：从 Queue 中取出一个已经处理好的 Dataset
        """
        # print("Waiting for next dataset from queue...")
        # get() 默认是阻塞的。如果后台线程够快，这里瞬间返回。
        next_ds = self.data_queue.get()
        
        self.dataset = next_ds
        if self.dataset is not None:
            self.current_file_index += 1
            log_with_file(f"Data file loaded successfully. Dataset size: {len(self.dataset.samples)} samples", args=self.args)
        else:
            print("All files processed in this epoch (Queue returned None).")

    def get_samples(self, num_samples):
        """
        获取 num_samples 个样本。
        如果当前 Dataset 不够，会自动从 Queue 拿下一个。
        """
        prompt_samples = []
        remaining_samples = num_samples

        while remaining_samples > 0:
            if self.dataset is not None:
                current_len = len(self.dataset.samples)
                
                # 情况 A: 当前文件剩下的样本足够多
                if self.sample_offset + remaining_samples <= current_len:
                    prompt_samples += self.dataset.samples[self.sample_offset: self.sample_offset + remaining_samples]
                    self.sample_offset += remaining_samples
                    remaining_samples = 0 
                
                # 情况 B: 当前文件不够了，把剩下的全拿走，然后切下一个文件
                else:
                    remaining_in_file = current_len - self.sample_offset
                    prompt_samples += self.dataset.samples[self.sample_offset:]
                    remaining_samples -= remaining_in_file
                    self.sample_offset = current_len 

                    # === 切换文件 (极快) ===
                    self.load_next_file_from_queue()
                    
                    if self.dataset is None:
                        # 队列里没数据了 (None)，说明所有文件都跑完了
                        break 

                    self.sample_offset = 0
            
            # 如果 Dataset 也没了，且还需要样本，跳出循环
            if len(prompt_samples) < num_samples:
                if self.dataset is None:
                    break
                continue

        # === Epoch 结束的处理 ===
        if self.dataset is None:
            self.epoch_id += 1
            self.current_file_index = 0
            # 重启后台线程，准备下一轮数据
            self._restart_loader_thread()

        # === 构造最终返回格式 (List of Lists) ===
        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                # 这里 prompt_sample 已经是包含 tokens/loss_mask 的对象了
                # 使用 shallow copy 即可，因为 tokens 列表通常不会被修改
                sample = copy.copy(prompt_sample) 
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
            
        return samples
    
    def _restart_loader_thread(self):
        """当一个 epoch 结束后，重启线程从头读取文件"""
        # 1. 清空队列中可能残留的数据
        with self.data_queue.mutex:
            self.data_queue.queue.clear()
        
        # 2. 重新创建并启动线程，从 index 0 开始
        self.loader_thread = threading.Thread(
            target=self._data_loader_daemon, 
            args=(0,), 
            daemon=True
        )
        self.loader_thread.start()
        
        # 3. 主线程加载第一个
        self.load_next_file_from_queue()

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        # 注意：这里的 current_file_index 是主线程已经处理完(或正在处理)的文件数
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
        if not self.args.rollout_global_dataset or self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            print(f"Checkpoint {path} does not exist.")
            return

        print(f"Loading metadata from {path}")
        state_dict = torch.load(path)
        
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        
        # 恢复文件进度
        saved_file_index = state_dict.get("current_file_index", 0)
        # 实际上我们要恢复的是 saved_file_index 这个文件，因为 offset 是基于它的
        # 如果 offset 很大，说明这个文件之前可能还没读完
        # 但为了简单，我们的逻辑是：current_file_index 总是指向"下一个要被完全消耗的文件"
        # 这里需要根据你的保存逻辑微调，假设保存的是"正在处理的文件索引"
        
        resume_index = max(0, saved_file_index - 1) if saved_file_index > 0 else 0
        
        # === 重置异步线程 ===
        # 1. 停止旧线程 (发送信号)
        self.stop_event.set()
        
        # 2. 清空队列
        with self.data_queue.mutex:
            self.data_queue.queue.clear()
        
        self.stop_event.clear()

        # 3. 启动新线程，从断点开始
        print(f"[Resume] Restarting async loader from file index {resume_index}")
        self.loader_thread = threading.Thread(
            target=self._data_loader_daemon, 
            args=(resume_index,), 
            daemon=True
        )
        self.loader_thread.start()
        
        # 4. 加载当前文件
        self.load_next_file_from_queue()
        
        # 修正状态
        self.current_file_index = resume_index + 1
        # sample_offset 已经从 dict 恢复了，直接生效

# 继承类保持不变，只需确保 super() 调用正确
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

        if num_samples == 0:
            return samples

        # 调用父类的 get_samples 获取更多数据
        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []
        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        if not samples: return
        for i in range(len(samples)):
            self.buffer.append(samples[i])

    def get_buffer_length(self):
        return len(self.buffer)

def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples

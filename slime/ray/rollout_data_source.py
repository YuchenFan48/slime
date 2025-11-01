import copy
import os
from pathlib import Path
import glob
import time

import torch
from transformers import AutoTokenizer

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.types import Sample


# TODO may further refactor data-loading part later
class RolloutDataSource:
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.data_files = []
        # TODO remove this
        self.metadata = {}

        # Check if the prompt_data is a directory and find all .jsonl and .parquet files
        if os.path.isdir(args.prompt_data):
            self.data_files = glob.glob(os.path.join(args.prompt_data, "*.jsonl")) + \
                              glob.glob(os.path.join(args.prompt_data, "*.parquet"))
        else:
            self.data_files = [args.prompt_data]  # If it's not a directory, treat it as a single file

        self.data_files = sorted([os.path.abspath(f) for f in self.data_files])
        self.dataset = None
        self.current_file_index = 0
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        if args.rollout_global_dataset:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")

        self.load_next_file(self.tokenizer)


    def load_next_file(self, tokenizer):
        """Load the next file and initialize the dataset"""
        if self.current_file_index < len(self.data_files):
            file_path = self.data_files[self.current_file_index]
            print(f"file path:{file_path}")
            self.dataset = Dataset(
                file_path,
                tokenizer=self.tokenizer,
                max_length=self.args.rollout_max_prompt_len,
                prompt_key=self.args.input_key,
                label_key=self.args.label_key,
                metadata_key=self.args.metadata_key,
                tool_key=self.args.tool_key,
                apply_chat_template=self.args.apply_chat_template,
                seed=self.args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            self.current_file_index += 1
        else:
            self.dataset = None
        print(f"self.dataset size: {len(self.dataset.samples)}")

    def get_samples(self, num_samples):
        # TODO further improve code
        start = time.time()
        prompt_samples = []
        remaining_samples = num_samples

        while remaining_samples > 0:
            if self.dataset is not None:
                # Check if enough samples can be taken from the current dataset
                if self.sample_offset + remaining_samples <= len(self.dataset.samples):
                    prompt_samples += self.dataset.samples[self.sample_offset: self.sample_offset + remaining_samples]
                    self.sample_offset += remaining_samples
                    remaining_samples = 0  # All requested samples have been taken
                else:
                    # Take all remaining samples from the current file
                    remaining_in_file = len(self.dataset.samples) - self.sample_offset
                    prompt_samples += self.dataset.samples[self.sample_offset:]
                    remaining_samples -= remaining_in_file
                    self.sample_offset = len(self.dataset.samples)  # All samples from the current file are consumed

                    # If the current file is exhausted, load the next file
                    if self.sample_offset >= len(self.dataset.samples):
                        # Load the next file if available
                        self.load_next_file(self.tokenizer)
                        if self.dataset is None:
                            break  # No more files to process

                        # Shuffle the dataset for the new file if required
                        if self.args.rollout_shuffle:
                            self.dataset.shuffle(self.epoch_id)

                        self.sample_offset = 0
            # If we run out of samples in the current file and still need more, continue processing files
            if len(prompt_samples) < num_samples:
                continue  # Go to the next iteration to process additional files if needed

        # After all files in the current epoch are processed, increment epoch_id
        if self.current_file_index >= len(self.data_files):
            self.epoch_id += 1
            self.current_file_index = 0  # Reset file index to start from the first file in the next epoch

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        end = time.time()
        print(f"Time need: {end - start}")
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
            "current_file_index": self.current_file_index,  # 新增
            "current_file_path": self.data_files[self.current_file_index - 1] if self.current_file_index > 0 else None,  # 新增
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)


    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            print(f"Checkpoint {path} does not exist.")
            return

        print(f"load metadata from {path}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        self.current_file_index = state_dict.get("current_file_index", 0)  # 新增
        current_file_path = state_dict.get("current_file_path", None)  # 新增

        # 重新加载数据集到正确的文件位置
        if current_file_path and current_file_path in self.data_files:
            self.current_file_index = self.data_files.index(current_file_path)
            self.load_next_file(self.tokenizer)
        else:
            self.current_file_index = 0
            self.load_next_file(self.tokenizer)
    
        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples

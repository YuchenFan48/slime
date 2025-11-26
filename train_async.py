import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary

import wandb
from datetime import datetime
import os

# === 修复开始：注册自定义 Config 和 Tokenizer ===
import sys
from transformers import AutoConfig, AutoTokenizer, Qwen2Config

# 尝试导入 Qwen2 的 Tokenizer，如果版本太旧可能需要 fallback

from transformers import Qwen2Tokenizer, Qwen2TokenizerFast


class Qwen3KimiConfig(Qwen2Config):
    model_type = "qwen3_kimi"

    def __init__(
        self,
        linear_conv_kernel_dim=4,
        linear_num_value_heads=None,
        linear_num_key_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        attention_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.attention_bias = attention_bias

# 1. 注册 Config

AutoConfig.register("qwen3_kimi", Qwen3KimiConfig)
print("[Slime] Successfully registered 'qwen3_kimi' config.")


AutoTokenizer.register(Qwen3KimiConfig, slow_tokenizer_class=Qwen2Tokenizer, fast_tokenizer_class=Qwen2TokenizerFast)
print("[Slime] Successfully registered 'qwen3_kimi' tokenizer mapping.")
# === 修复结束 ===
# Global variable for default log file path
_default_log_file = None
def log_with_file(message, log_file=None, args=None):
    """Log message to both console and file with timestamp."""
    global _default_log_file
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    
    # Get log file path from args if not specified
    if log_file is None:
        if args is not None and hasattr(args, 'log_file_path') and args.log_file_path is not None:
            log_file = args.log_file_path
        else:
            # Create default path with timestamp (once per program run)
            if _default_log_file is None:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                _default_log_file = f"training_metrics_{timestamp_str}.log"
            log_file = _default_log_file
    
    # Ensure log directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) if os.path.dirname(log_file) else ".", exist_ok=True)
    except OSError:
        # If directory creation fails (e.g., disk quota exceeded), skip file logging
        return
    
    # Append to log file
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    except OSError as e:
        # If file write fails (e.g., disk quota exceeded), print warning but don't crash
        print(f"Warning: Failed to write to log file {log_file}: {e}")
        print("Continuing training without file logging...")

def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager, wandb_run_id=wandb_run_id)

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        with open('log.log', 'a') as f:
            f.write(f"Step: {rollout_id}")
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            actor_model.save_model(rollout_id)
            if args.use_critic:
                critic_model.save_model(rollout_id)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            # 1. 获取数据集字典
            dataset_dict = ray.get(rollout_manager.eval.remote(rollout_id))

            # 2. 顺序地对每个数据集进行评估
            all_results = {}
            for name, dataset in dataset_dict.items():
                # 调用并等待当前数据集的评估完成，然后再处理下一个
                logs = ray.get(actor_model.async_eval(rollout_id, dataset))
                all_results[name] = logs

            # 3. 聚合和格式化日志数据
            final_log_dict = {}
            
            # 假设至少有一个数据集有结果，并且所有结果的 'eval/step' 都相同
            if all_results:
                first_dataset_key = next(iter(all_results))
                if all_results[first_dataset_key]: # 确保第一个结果列表不为空
                    final_log_dict['eval/step'] = all_results[first_dataset_key][0]['eval/step']

            # 遍历每个数据集的结果，计算其平均 PPL
            all_ppls = []

            for name, logs in all_results.items():
                if not logs:  # 如果某个数据集没有返回任何日志，则跳过
                    continue
                
                ppls = [log['eval/ppl'] for log in logs]
                avg_ppl = sum(ppls) / len(ppls)
                
                # 保存每个数据集的平均 PPL
                final_log_dict[f'eval/{name}'] = avg_ppl
                
                # 将当前数据集的所有 PPL 值加入全局列表
                all_ppls.extend(ppls)

            # 计算并保存所有数据源的总体平均 PPL
            if all_ppls:
                global_avg_ppl = sum(all_ppls) / len(all_ppls)
                final_log_dict['eval/all'] = global_avg_ppl

            print(final_log_dict)
            log_with_file(f"eval {rollout_id}: {final_log_dict}", args=args)
            # 4. 将聚合后的结果记录到 Wandb
            if final_log_dict:
                wandb.log(final_log_dict)

        

if __name__ == "__main__":
    args = parse_args()
    train(args)

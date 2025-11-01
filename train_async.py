import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary

import wandb
from datetime import datetime
import os

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
    os.makedirs(os.path.dirname(os.path.abspath(log_file)) if os.path.dirname(log_file) else ".", exist_ok=True)
    
    # Append to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

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

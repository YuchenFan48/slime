import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action

# Register Kimi model config and tokenizer
import slime.kimi


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

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

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            # 1. 获取数据集字典 (用于 PPL 评估)
            dataset_dict = ray.get(rollout_manager.eval.remote(rollout_id))

            # 2. 如果有评估数据，顺序地对每个数据集进行评估
            if dataset_dict is not None:
                all_results = {}
                for name, dataset in dataset_dict.items():
                    # 调用并等待当前数据集的评估完成，然后再处理下一个
                    logs = ray.get(actor_model.async_eval(rollout_id, dataset))
                    all_results[name] = logs

                # 3. 聚合和格式化日志数据
                final_log_dict = {}
                
                # 获取 eval/step
                if all_results:
                    first_dataset_key = next(iter(all_results))
                    if all_results[first_dataset_key]:
                        final_log_dict['eval/step'] = all_results[first_dataset_key][0].get('eval/step', rollout_id)

                # 遍历每个数据集的结果，计算其平均 PPL
                all_ppls = []

                for name, logs in all_results.items():
                    if not logs:
                        continue
                    
                    ppls = [log['eval/ppl'] for log in logs if 'eval/ppl' in log]
                    if ppls:
                        avg_ppl = sum(ppls) / len(ppls)
                        final_log_dict[f'eval/{name}'] = avg_ppl
                        all_ppls.extend(ppls)

                # 计算并保存所有数据源的总体平均 PPL
                if all_ppls:
                    global_avg_ppl = sum(all_ppls) / len(all_ppls)
                    final_log_dict['eval/all'] = global_avg_ppl

                # 4. 将聚合后的结果记录
                if final_log_dict:
                    from slime.utils import logging_utils
                    logging_utils.log(args, final_log_dict, step_key='eval/step')

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)

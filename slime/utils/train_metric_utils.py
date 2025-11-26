from argparse import Namespace
from copy import deepcopy
from typing import Callable

import wandb

from slime.utils.timer import Timer

from datetime import datetime
import os

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

def log_perf_data_raw(
    rollout_id: int, args: Namespace, is_primary_rank: bool, compute_total_fwd_flops: Callable
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    log_with_file(f"perf {rollout_id}: {log_dict}", args=args)

    step = (
        rollout_id
        if not args.wandb_always_use_train_step
        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )
    if args.use_wandb:
        log_dict["rollout/step"] = step
        wandb.log(log_dict)

    if args.use_tensorboard:
        from slime.utils.tensorboard_utils import _TensorboardAdapter

        tb = _TensorboardAdapter(args)
        tb.log(data=log_dict, step=step)

import os
import wandb


from functools import wraps

def with_proxy(proxy_url="http://hk-mmhttpproxy.woa.com:11113/"):
    """装饰器：临时设置代理"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 保存原始环境变量
            original_env = {
                'http_proxy': os.environ.get('http_proxy'),
                'https_proxy': os.environ.get('https_proxy'),
                'all_proxy': os.environ.get('all_proxy'),
                'HTTP_PROXY': os.environ.get('HTTP_PROXY'),
                'HTTPS_PROXY': os.environ.get('HTTPS_PROXY'),
                'ALL_PROXY': os.environ.get('ALL_PROXY'),
            }
            
            try:
                # 设置代理
                os.environ['http_proxy'] = proxy_url
                os.environ['https_proxy'] = proxy_url
                os.environ['all_proxy'] = proxy_url
                os.environ['HTTP_PROXY'] = proxy_url
                os.environ['HTTPS_PROXY'] = proxy_url
                os.environ['ALL_PROXY'] = proxy_url
                
                print(f"Proxy enabled for {func.__name__}: {proxy_url}")
                
                # 执行函数
                result = func(*args, **kwargs)
                return result
                
            finally:
                # 恢复原始环境变量
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                print(f"Proxy settings restored after {func.__name__}")
        
        return wrapper
    return decorator


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"

@with_proxy()
def init_wandb_primary(args):
    if not args.use_wandb:
        return None

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            print("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            print("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            print("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": args.__dict__,
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        print(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    return wandb.run.id


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
@with_proxy()
def init_wandb_secondary(args, wandb_run_id, router_addr=None):
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    if args.sglang_enable_metrics and router_addr is not None:
        print(f"Forward SGLang metrics at {router_addr} to WandB.")
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": args.__dict__,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()

@with_proxy()
def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")


def get_wandb_offline_dir(args):
    """Get the directory where offline W&B data is stored."""
    if _is_offline_mode(args):
        if args and hasattr(args, "wandb_dir") and args.wandb_dir:
            # Use custom directory if specified
            return args.wandb_dir
        else:
            # Default offline directory is ~/wandb/offline-run-<timestamp>
            # This will be created automatically by wandb
            return os.path.expanduser("~/wandb")
    return None

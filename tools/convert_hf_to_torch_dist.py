import json
import os
import shutil

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

import slime_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from slime.backends.megatron_utils import set_default_megatron_args
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func

from mbridge.core.auto_bridge import AutoBridge
from transformers import AutoConfig, Qwen2Config

# 1. 定义 Qwen3KimiConfig
# 既然是 Qwen3 改版，通常继承自 Qwen2Config 是最方便的，
# 然后把你在 KimiDeltaAttention 中用到的新参数加进去。
class Qwen3KimiConfig(Qwen2Config):
    model_type = "qwen3_kimi"

    def __init__(
        self,
        linear_conv_kernel_dim=4,
        linear_num_value_heads=None,
        linear_num_key_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim

# 2. 定义 Qwen3NextInfLLMV2Config
class Qwen3NextInfLLMV2Config(Qwen2Config):
    model_type = "qwen3nextinfllm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # InfLLM V2 使用标准的 Qwen3 Next 配置，不需要额外参数
        # 权重格式的区别由 bridge 处理

# 3. 注册这些 Config
# 这一步告诉 transformers：当遇到对应的 "model_type" 时，使用对应的 Config 类
try:
    AutoConfig.register("qwen3_kimi", Qwen3KimiConfig)
except ValueError:
    # 防止重复注册报错
    pass

try:
    AutoConfig.register("qwen3nextinfllm", Qwen3NextInfLLMV2Config)
except ValueError:
    # 防止重复注册报错
    pass


def add_convertion_args(parser):
    """Add conversion arguments to the parser"""
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--model-type", type=str, default=None, 
                       help="Override model type for bridge selection (e.g., 'qwen3nextinfllm'). "
                            "If not set, will use model_type from config.json")
    
    # InfLLM V2 参数（用于模型架构构建）
    parser.add_argument(
        "--infllmv2-topk-blocks",
        type=int,
        default=64,
        help="Number of top-k blocks to select in InfLLM V2 Stage 1",
    )
    parser.add_argument(
        "--infllmv2-block-size",
        type=int,
        default=64,
        help="Block size for InfLLM V2 sparse attention",
    )
    parser.add_argument(
        "--infllmv2-use-stage1",
        type=lambda x: x.lower() in ('true', '1', 'yes') if isinstance(x, str) else bool(x),
        default=True,
        help="Enable Stage 1 for top-k selection in InfLLM V2",
    )
    parser.add_argument(
        "--infllmv2-use-for-linear-attention",
        type=lambda x: x.lower() in ('true', '1', 'yes') if isinstance(x, str) else bool(x),
        default=False,
        help="Whether to use InfLLM V2 for linear attention layers (not recommended)",
    )
    
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except:
        pass
    return parser


def get_args():
    args = parse_args(add_convertion_args)
    args = set_default_megatron_args(args)

    # set to pass megatron validate_args
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}. "
        "You are using to much GPUs for this conversion."
    )

    ceildiv = lambda a, b: -(a // -b)  # Ceiling division

    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)

            if args.decoder_last_pipeline_num_layers > 0:
                break

            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
                )
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )

    validate_args(args)
    return args


def main():
    """Initialize distributed environment"""
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    args = get_args()
    init(args)
    # 强制读取 HF Config
    # Load model
    hf_model_path = args.hf_checkpoint
    config_json_path = os.path.join(hf_model_path, "config.json")
    
    # 如果指定了 --model-type，临时修改 config.json 的 model_type 以选择正确的 bridge
    original_model_type = None
    config_backup = None
    if args.model_type:
        # 读取并备份原始 config
        with open(config_json_path, "r") as f:
            config_backup = json.load(f)
        original_model_type = config_backup.get("model_type")
        
        # 修改 model_type
        config_backup["model_type"] = args.model_type
        with open(config_json_path, "w") as f:
            json.dump(config_backup, f, indent=2)
        print(f"Temporarily modified config.json: model_type changed from '{original_model_type}' to '{args.model_type}'")
    
    try:
        hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        
        # 2. 覆盖通用参数以匹配
        args.hidden_size = getattr(hf_config, "hidden_size", args.hidden_size)
        args.num_attention_heads = getattr(hf_config, "num_attention_heads", args.num_attention_heads)
        model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

        # AutoBridge 会根据 config.json 的 model_type 自动选择 bridge
        bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    finally:
        # 恢复原始 config.json
        if config_backup and original_model_type:
            config_backup["model_type"] = original_model_type
            with open(config_json_path, "w") as f:
                json.dump(config_backup, f, indent=2)
            print(f"Restored config.json: model_type changed back to '{original_model_type}'")
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"Model loaded: {hf_model_path}")
    print(model)
    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        # change to release ckpt
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write("release")
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

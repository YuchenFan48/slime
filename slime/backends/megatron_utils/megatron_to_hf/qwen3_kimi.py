import re
import torch
from packaging.version import parse

# 尝试导入 sglang 以检查版本
try:
    import sglang
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

def _convert_layer_internal(args, layer_prefix, rest, param):
    """
    通用层参数转换函数。
    将 Megatron-Core 的层内部参数名映射为标准 HF 格式。
    
    Args:
        args: 配置参数
        layer_prefix: HF 层的层级前缀，例如 "model.layers.0" 或 "mtp.layers.0.transformer_layer"
        rest: 去除层级前缀后的剩余参数名，例如 "self_attention.input_layernorm.weight"
        param: 参数 Tensor
    """
    
    # 1. 计算 Head Dim (用于 QKV 切分)
    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # === [LayerNorms] 处理层归一化 ===
    # 注意：这是最容易出错的地方，Mcore 可能将 Input LN 放在不同位置
    
    # 情况 A: Fused LayerNorm (通常在 linear_qgkv 中)
    if rest == "self_attention.linear_qgkv.layer_norm_weight":
        return [(f"{layer_prefix}.input_layernorm.weight", param)]
    
    # 情况 B: 独立的 Input LayerNorm (Kimi 架构中常见，位于 Attention 模块内)
    if rest == "self_attention.input_layernorm.weight":
        return [(f"{layer_prefix}.input_layernorm.weight", param)]
    
    # 情况 C: Post Attention / Pre MLP LayerNorm
    if rest == "mlp.linear_fc1.layer_norm_weight" or rest == "pre_mlp_layernorm.weight":
        return [(f"{layer_prefix}.post_attention_layernorm.weight", param)]
    
    # 情况 D: QK LayerNorm (Attention 内部)
    if rest == "self_attention.q_layernorm.weight":
        return [(f"{layer_prefix}.self_attn.q_norm.weight", param)]
    if rest == "self_attention.k_layernorm.weight":
        return [(f"{layer_prefix}.self_attn.k_norm.weight", param)]

    # === [Attention] Kimi 特有的 Linear Attention ===
    # 匹配 self_attention.linear_attn.xxx
    if rest.startswith("self_attention.linear_attn."):
        sub_name = rest[len("self_attention.linear_attn.") :]
        # 直接映射: q_proj, k_proj, v_proj, o_proj, q/k/v_conv1d, f_a/b_proj, A_log, dt_bias 等
        return [(f"{layer_prefix}.self_attn.{sub_name}", param)]

    # === [Attention] 标准 Self Attention (Full Attention) ===
    # 处理 QKV 权重切分 (Megatron 格式 -> HF 格式)
    if rest == "self_attention.linear_qgkv.weight":
        param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
        q_param, k_param, v_param = torch.split(
            param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
        )
        q_param = (
            q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
            .transpose(1, 2)
            .reshape(-1, args.hidden_size)
        )
        k_param = k_param.reshape(-1, args.hidden_size)
        v_param = v_param.reshape(-1, args.hidden_size)
        return [
            (f"{layer_prefix}.self_attn.q_proj.weight", q_param),
            (f"{layer_prefix}.self_attn.k_proj.weight", k_param),
            (f"{layer_prefix}.self_attn.v_proj.weight", v_param),
        ]
    
    # 处理 QKV Bias
    if rest == "self_attention.linear_qgkv.bias":
        param = param.view(args.num_query_groups, -1)
        q_bias, k_bias, v_bias = torch.split(
            param,
            split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
            dim=1,
        )
        q_bias = q_bias.contiguous().flatten()
        k_bias = k_bias.contiguous().flatten()
        v_bias = v_bias.contiguous().flatten()
        return [
            (f"{layer_prefix}.self_attn.q_proj.bias", q_bias),
            (f"{layer_prefix}.self_attn.k_proj.bias", k_bias),
            (f"{layer_prefix}.self_attn.v_proj.bias", v_bias),
        ]

    # 处理 Output Projection
    if rest == "self_attention.linear_proj.weight":
        return [(f"{layer_prefix}.self_attn.o_proj.weight", param)]

    # 兜底：处理 self_attention.self_attn.xxx 这种嵌套
    if rest.startswith("self_attention.self_attn."):
        sub_name = rest[len("self_attention.self_attn.") :]
        return [(f"{layer_prefix}.self_attn.{sub_name}", param)]

    # === [MoE Experts] 处理专家层 ===
    # 匹配模式: mlp.experts.linear_fc1.weight1 或 mlp.experts.linear_fc1.weight01
    expert_pattern = r"mlp\.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest_expert, expert_idx = match.groups()
        expert_idx = int(expert_idx)
        
        if rest_expert == "linear_fc1":
            # Gate / Up Projection
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{layer_prefix}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                (f"{layer_prefix}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
            ]
        elif rest_expert == "linear_fc2":
            # Down Projection
            outputs = [
                (f"{layer_prefix}.mlp.experts.{expert_idx}.down_proj.weight", param),
            ]
            # SGLang MoE 特有的 scale 参数兼容
            if SGLANG_AVAILABLE and parse(sglang.__version__) < parse("0.4.9.post5") and getattr(args, 'sglang_enable_ep_moe', False):
                outputs += [
                    (
                        f"{layer_prefix}.mlp.experts.{expert_idx}.down_proj.input_scale",
                        torch.tensor(1.0, dtype=torch.float32, device=param.device),
                    ),
                    (
                        f"{layer_prefix}.mlp.experts.{expert_idx}.down_proj.weight_scale",
                        torch.tensor(1.0, dtype=torch.float32, device=param.device),
                    ),
                ]
            return outputs

    # === [Shared Experts] 处理共享专家 ===
    shared_expert_pattern = r"mlp\.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest_shared = match.groups()[0]
        if rest_shared == "linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{layer_prefix}.mlp.shared_expert.gate_proj.weight", gate_weight),
                (f"{layer_prefix}.mlp.shared_expert.up_proj.weight", up_weight),
            ]
        elif rest_shared == "linear_fc2.weight":
            return [(f"{layer_prefix}.mlp.shared_expert.down_proj.weight", param)]
        elif rest_shared == "gate_weight":
            return [(f"{layer_prefix}.mlp.shared_expert_gate.weight", param)]

    # === [Standard MLP] 处理普通 MLP (Dense) ===
    if rest == "mlp.linear_fc1.weight":
        gate_weight, up_weight = param.chunk(2, dim=0)
        return [
            (f"{layer_prefix}.mlp.gate_proj.weight", gate_weight),
            (f"{layer_prefix}.mlp.up_proj.weight", up_weight),
        ]
    elif rest == "mlp.linear_fc2.weight":
        return [(f"{layer_prefix}.mlp.down_proj.weight", param)]
    
    # === [Router / Gate] ===
    elif rest == "mlp.router.weight":
        return [(f"{layer_prefix}.mlp.gate.weight", param)]
    elif rest == "mlp.router.expert_bias":
        return [(f"{layer_prefix}.mlp.gate.e_score_correction_bias", param)]

    return None

def convert_qwen3_kimi_to_hf(args, name, param):
    """
    转换入口函数
    """
    # 1. Embeddings & Heads & Global Norms
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]
    # MTP Global Norm
    if name == "module.module.mtp.final_layernorm.weight":
         return [("model.mtp.norm.weight", param)] 

    # 2. Main Decoder Layers
    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        # 对于 Decoder 层，HF 路径通常是 model.layers.N...
        layer_prefix = f"model.layers.{layer_idx}"
        
        outputs = _convert_layer_internal(args, layer_prefix, rest, param)
        if outputs is not None:
            return outputs
        # 如果内部没处理，抛出异常
        raise ValueError(f"Unknown parameter name in decoder layer {layer_idx}: {rest} (Full: {name})")

    # 3. MTP Layers (Multi-Task Prediction)
    mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        # 对于 MTP 层，HF 路径通常是 mtp.layers.N...
        layer_prefix = f"mtp.layers.{layer_idx}"
        
        # === MTP 模块级特定参数 ===
        if rest == "enorm.weight":
            return [(f"{layer_prefix}.enorm.weight", param)]
        elif rest == "hnorm.weight":
            return [(f"{layer_prefix}.hnorm.weight", param)]
        elif rest == "eh_proj.weight":
            return [(f"{layer_prefix}.eh_proj.weight", param)]
        elif rest == "final_layernorm.weight":
            return [(f"{layer_prefix}.final_layernorm.weight", param)]

        # === Transformer Block 内部参数 ===
        # MTP 内部通常包裹了一个 transformer_layer
        if rest.startswith("transformer_layer."):
            inner_rest = rest[len("transformer_layer."):]
            # HF 路径需要带上 transformer_layer 以区分
            inner_prefix = f"{layer_prefix}.transformer_layer"
            outputs = _convert_layer_internal(args, inner_prefix, inner_rest, param)
            if outputs is not None:
                return outputs
        else:
            # 某些实现可能直接放在 mtp layer 下
            outputs = _convert_layer_internal(args, layer_prefix, rest, param)
            if outputs is not None:
                return outputs

        raise ValueError(f"Unknown parameter name in MTP layer {layer_idx}: {rest} (Full: {name})")

    # 如果所有规则都不匹配
    raise ValueError(f"Unknown parameter name: {name}")

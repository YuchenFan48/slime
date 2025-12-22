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
        rest: 去除层级前缀后的剩余参数名，例如 "self_attention.query_key_value.weight"
        param: 参数 Tensor
    """
    
    # 1. 计算 Head Dim (用于 QKV 切分)
    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # === [LayerNorms] 处理层归一化 ===
    
    # 情况 A: Fused LayerNorm (通常在 linear_qgkv 中)
    if rest == "self_attention.linear_qgkv.layer_norm_weight":
        return [(f"{layer_prefix}.input_layernorm.weight", param)]
    
    # 情况 B: 独立的 Input LayerNorm (InfLLM V2 架构中，位于 Attention 模块内)
    if rest == "self_attention.input_layernorm.weight":
        return [(f"{layer_prefix}.input_layernorm.weight", param)]
    
    # 情况 C: Post Attention / Pre MLP LayerNorm
    if rest == "mlp.linear_fc1.layer_norm_weight" or rest == "pre_mlp_layernorm.weight":
        return [(f"{layer_prefix}.post_attention_layernorm.weight", param)]
    
    # 情况 D: QK LayerNorm (InfLLM V2 Attention 内部，如果启用)
    if rest == "self_attention.q_layernorm.weight":
        return [(f"{layer_prefix}.self_attn.q_norm.weight", param)]
    if rest == "self_attention.k_layernorm.weight":
        return [(f"{layer_prefix}.self_attn.k_norm.weight", param)]

    # === [Attention] InfLLM V2 Attention ===
    # InfLLM V2 使用标准的 QKV projection，需要切分
    # InfLLMV2Attention 使用 query_key_value (ColumnParallelLinear)
    # 格式: Q 在前 (num_attention_heads * head_dim)，然后是 K (num_query_groups * head_dim)，最后是 V (num_query_groups * head_dim)
    # 总大小: (num_attention_heads + 2 * num_query_groups) * head_dim
    # 
    # 注意：在 tensor parallel 的情况下，参数可能已经被合并（完整大小），也可能是单个 partition（需要合并）
    # 这里假设参数已经被合并，如果是单个 partition，需要在转换前先合并所有 TP rank 的参数
    
    # 处理 QKV 权重切分 (Megatron 格式 -> HF 格式)
    if rest == "self_attention.query_key_value.weight":
        # 计算完整的 QKV 维度
        full_q_dim = args.num_attention_heads * head_dim
        full_kv_dim = args.num_query_groups * head_dim
        full_qkv_dim = full_q_dim + 2 * full_kv_dim
        
        # 检查参数形状，判断是否需要处理 TP 切分
        # 如果参数已经被合并，形状应该是 (full_qkv_dim, hidden_size)
        # 如果是单个 partition，形状可能是 (full_qkv_dim // tp_size, hidden_size)
        param_qkv_dim = param.shape[0]
        
        # 如果参数形状匹配完整大小，直接切分
        if param_qkv_dim == full_qkv_dim:
            q_dim = full_q_dim
            kv_dim = full_kv_dim
        else:
            # 如果参数是单个 partition，需要根据实际形状推断
            # 假设参数已经被正确合并，如果形状不匹配，可能需要警告
            # 这里假设参数已经被合并，如果形状不匹配，使用实际形状
            # 注意：这种情况通常不应该发生，如果发生可能需要检查转换流程
            q_dim = args.num_attention_heads * head_dim
            kv_dim = args.num_query_groups * head_dim
            if param_qkv_dim != q_dim + 2 * kv_dim:
                # 如果形状不匹配，尝试根据实际形状推断
                # 这可能是 TP 切分的情况，但通常应该在转换前合并
                raise ValueError(
                    f"Unexpected param shape for query_key_value.weight: "
                    f"expected {(full_qkv_dim, args.hidden_size)}, got {param.shape}. "
                    f"This might indicate TP parameters need to be merged first."
                )
        
        # 切分 Q, K, V（按照顺序：Q -> K -> V）
        q_param = param[:q_dim, :]  # (num_heads * head_dim, hidden_size)
        k_param = param[q_dim:q_dim + kv_dim, :]  # (num_kv_heads * head_dim, hidden_size)
        v_param = param[q_dim + kv_dim:, :]  # (num_kv_heads * head_dim, hidden_size)
        
        return [
            (f"{layer_prefix}.self_attn.q_proj.weight", q_param),
            (f"{layer_prefix}.self_attn.k_proj.weight", k_param),
            (f"{layer_prefix}.self_attn.v_proj.weight", v_param),
        ]
    
    # 处理 QKV Bias（如果启用）
    if rest == "self_attention.query_key_value.bias":
        # 计算完整的 QKV 维度
        full_q_dim = args.num_attention_heads * head_dim
        full_kv_dim = args.num_query_groups * head_dim
        full_qkv_dim = full_q_dim + 2 * full_kv_dim
        
        # 检查参数形状
        param_qkv_dim = param.shape[0]
        
        if param_qkv_dim == full_qkv_dim:
            q_dim = full_q_dim
            kv_dim = full_kv_dim
        else:
            q_dim = args.num_attention_heads * head_dim
            kv_dim = args.num_query_groups * head_dim
            if param_qkv_dim != q_dim + 2 * kv_dim:
                raise ValueError(
                    f"Unexpected param shape for query_key_value.bias: "
                    f"expected {full_qkv_dim}, got {param.shape}. "
                    f"This might indicate TP parameters need to be merged first."
                )
        
        # 切分 Q, K, V bias（按照顺序：Q -> K -> V）
        q_bias = param[:q_dim]  # (num_heads * head_dim,)
        k_bias = param[q_dim:q_dim + kv_dim]  # (num_kv_heads * head_dim,)
        v_bias = param[q_dim + kv_dim:]  # (num_kv_heads * head_dim,)
        
        return [
            (f"{layer_prefix}.self_attn.q_proj.bias", q_bias),
            (f"{layer_prefix}.self_attn.k_proj.bias", k_bias),
            (f"{layer_prefix}.self_attn.v_proj.bias", v_bias),
        ]

    # === [Attention] 标准 Self Attention (Full Attention) ===
    # 处理 QKV 权重切分 (Megatron 格式 -> HF 格式)
    # 兼容原有的 linear_qgkv 格式（用于非 InfLLM V2 的层）
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

    # 处理 Output Projection (InfLLM V2 使用 dense)
    if rest == "self_attention.dense.weight":
        return [(f"{layer_prefix}.self_attn.o_proj.weight", param)]
    if rest == "self_attention.dense.bias":
        return [(f"{layer_prefix}.self_attn.o_proj.bias", param)]
    
    # 处理 Output Projection (标准格式)
    if rest == "self_attention.linear_proj.weight":
        return [(f"{layer_prefix}.self_attn.o_proj.weight", param)]

    # 兜底：处理 self_attention.self_attn.xxx 这种嵌套
    if rest.startswith("self_attention.self_attn."):
        sub_name = rest[len("self_attention.self_attn.") :]
        return [(f"{layer_prefix}.self_attn.{sub_name}", param)]

    # === [Linear Attention] 处理 linear_attention 层（如果存在） ===
    # 匹配 self_attention.linear_attn.xxx
    # 注意：根据 qwen3_next.py 的实现，linear_attn 参数直接映射到 layer.linear_attn.xxx
    # 而不是 layer.self_attn.linear_attn.xxx（在HF模型中，linear_attn是直接作为layer的一个属性）
    if rest.startswith("self_attention.linear_attn."):
        sub_name = rest[len("self_attention.linear_attn.") :]
        # 直接映射: A_log, conv1d.weight, dt_bias, in_proj_ba.weight, in_proj_qkvz.weight, norm.weight, out_proj.weight 等
        # 参考 qwen3_next.py，这些参数应该直接映射到 layer.linear_attn.xxx
        return [(f"{layer_prefix}.linear_attn.{sub_name}", param)]

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

    # === [兜底机制] 处理其他可能的参数名格式 ===
    # 对于 InfLLM V2，可能存在的嵌套参数名：self_attention.infllmv2_attn.xxx
    if rest.startswith("self_attention.infllmv2_attn."):
        sub_name = rest[len("self_attention.infllmv2_attn.") :]
        # 将 infllmv2_attn.query_key_value -> self_attn.q_proj/k_proj/v_proj
        if sub_name == "query_key_value.weight":
            # 使用与上面相同的逻辑处理 TP 情况
            full_q_dim = args.num_attention_heads * head_dim
            full_kv_dim = args.num_query_groups * head_dim
            full_qkv_dim = full_q_dim + 2 * full_kv_dim
            param_qkv_dim = param.shape[0]
            
            if param_qkv_dim == full_qkv_dim:
                q_dim = full_q_dim
                kv_dim = full_kv_dim
            else:
                q_dim = args.num_attention_heads * head_dim
                kv_dim = args.num_query_groups * head_dim
                if param_qkv_dim != q_dim + 2 * kv_dim:
                    raise ValueError(
                        f"Unexpected param shape for infllmv2_attn.query_key_value.weight: "
                        f"expected {(full_qkv_dim, args.hidden_size)}, got {param.shape}"
                    )
            
            q_param = param[:q_dim, :]
            k_param = param[q_dim:q_dim + kv_dim, :]
            v_param = param[q_dim + kv_dim:, :]
            return [
                (f"{layer_prefix}.self_attn.q_proj.weight", q_param),
                (f"{layer_prefix}.self_attn.k_proj.weight", k_param),
                (f"{layer_prefix}.self_attn.v_proj.weight", v_param),
            ]
        elif sub_name == "query_key_value.bias":
            # 使用与上面相同的逻辑处理 TP 情况
            full_q_dim = args.num_attention_heads * head_dim
            full_kv_dim = args.num_query_groups * head_dim
            full_qkv_dim = full_q_dim + 2 * full_kv_dim
            param_qkv_dim = param.shape[0]
            
            if param_qkv_dim == full_qkv_dim:
                q_dim = full_q_dim
                kv_dim = full_kv_dim
            else:
                q_dim = args.num_attention_heads * head_dim
                kv_dim = args.num_query_groups * head_dim
                if param_qkv_dim != q_dim + 2 * kv_dim:
                    raise ValueError(
                        f"Unexpected param shape for infllmv2_attn.query_key_value.bias: "
                        f"expected {full_qkv_dim}, got {param.shape}"
                    )
            
            q_bias = param[:q_dim]
            k_bias = param[q_dim:q_dim + kv_dim]
            v_bias = param[q_dim + kv_dim:]
            return [
                (f"{layer_prefix}.self_attn.q_proj.bias", q_bias),
                (f"{layer_prefix}.self_attn.k_proj.bias", k_bias),
                (f"{layer_prefix}.self_attn.v_proj.bias", v_bias),
            ]
        elif sub_name == "dense.weight":
            return [(f"{layer_prefix}.self_attn.o_proj.weight", param)]
        elif sub_name == "dense.bias":
            return [(f"{layer_prefix}.self_attn.o_proj.bias", param)]
        elif sub_name == "q_layernorm.weight":
            return [(f"{layer_prefix}.self_attn.q_norm.weight", param)]
        elif sub_name == "k_layernorm.weight":
            return [(f"{layer_prefix}.self_attn.k_norm.weight", param)]
    
    # === [通用兜底] 处理其他直接映射的参数 ===
    # 参考 qwen3_next.py 的做法，对于某些可以直接映射的参数名
    # 注意：这里只处理不需要特殊转换的参数，需要切分或特殊处理的参数已经在上面处理了
    if rest.startswith("self_attention."):
        sub_name = rest[len("self_attention.") :]
        # 直接映射的参数列表（这些参数名在 Megatron 和 HF 中相同）
        # 参考 qwen3_next.py 中的处理方式
        direct_map_params = [
            "input_layernorm.weight",
            # linear attn 参数（已经在上面单独处理，这里作为兜底）
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_ba.weight",
            "linear_attn.in_proj_qkvz.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            # gated attn 参数（如果存在）
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
            # 其他可能的参数
            "linear_qkv.layer_norm_weight",
            "linear_qkv.weight",
        ]
        if sub_name in direct_map_params:
            # 对于 linear_attn 参数，直接映射到 layer.linear_attn.xxx
            if sub_name.startswith("linear_attn."):
                return [(f"{layer_prefix}.{sub_name}", param)]
            # 对于其他参数，直接映射
            return [(f"{layer_prefix}.{sub_name}", param)]
        
        # 对于 self_attn.xxx 格式的参数，直接映射到 self_attn.xxx
        if sub_name.startswith("self_attn."):
            return [(f"{layer_prefix}.{sub_name}", param)]

    return None

def convert_qwen3_next_infllmv2_to_hf(args, name, param):
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


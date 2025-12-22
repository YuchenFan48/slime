import re
import torch
from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


@register_model("qwen3nextinfllm")
class Qwen3NextInfLLMV2Bridge(Qwen2MoEBridge):
    _ATTENTION_MAPPING = (
        Qwen2MoEBridge._ATTENTION_MAPPING
        | {
            f"self_attention.{weight_name}": ["model.layers.{layer_number}." + weight_name]
            for weight_name in [
                "input_layernorm.weight",
                # === InfLLM V2 Attention Components ===
                # InfLLM V2 使用 query_key_value 格式，而不是分开的 q_proj/k_proj/v_proj
                # 但 HF 格式仍然是分开的，所以需要特殊处理
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.bias",
                "self_attn.o_proj.weight",
                "self_attn.o_proj.bias",
                # === Qwen3 Next Linear Attention Components (for non-infllmv2 layers) ===
                # 注意：根据 Megatron 转 HF 的脚本，linear_attn 参数映射到 layer.linear_attn.xxx
                # 而不是 layer.self_attn.linear_attn.xxx（在HF模型中，linear_attn是直接作为layer的一个属性）
                "linear_attn.A_log",
                "linear_attn.conv1d.weight",
                "linear_attn.dt_bias",
                "linear_attn.in_proj_ba.weight",
                "linear_attn.in_proj_qkvz.weight",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            ]
        }
        | {
            # InfLLM V2 使用 query_key_value 格式
            "self_attention.query_key_value.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
            "self_attention.query_key_value.weight": [
                "model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.layers.{layer_number}.self_attn.k_proj.weight",
                "model.layers.{layer_number}.self_attn.v_proj.weight",
            ],
            "self_attention.query_key_value.bias": [
                "model.layers.{layer_number}.self_attn.q_proj.bias",
                "model.layers.{layer_number}.self_attn.k_proj.bias",
                "model.layers.{layer_number}.self_attn.v_proj.bias",
            ],
            # 兼容 infllmv2_attn 命名格式
            "self_attention.infllmv2_attn.query_key_value.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
            "self_attention.infllmv2_attn.query_key_value.weight": [
                "model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.layers.{layer_number}.self_attn.k_proj.weight",
                "model.layers.{layer_number}.self_attn.v_proj.weight",
            ],
            "self_attention.infllmv2_attn.query_key_value.bias": [
                "model.layers.{layer_number}.self_attn.q_proj.bias",
                "model.layers.{layer_number}.self_attn.k_proj.bias",
                "model.layers.{layer_number}.self_attn.v_proj.bias",
            ],
            "self_attention.infllmv2_attn.dense.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
            "self_attention.infllmv2_attn.dense.bias": ["model.layers.{layer_number}.self_attn.o_proj.bias"],
            "self_attention.infllmv2_attn.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
            "self_attention.infllmv2_attn.k_layernorm.weight": ["model.layers.{layer_number}.self_attn.k_norm.weight"],
            # 兼容原有的 linear_qgkv 格式（用于非 InfLLM V2 的层）
            "self_attention.linear_qgkv.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
            "self_attention.linear_qgkv.weight": [
                "model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.layers.{layer_number}.self_attn.k_proj.weight",
                "model.layers.{layer_number}.self_attn.v_proj.weight",
            ],
            "self_attention.linear_qgkv.bias": [
                "model.layers.{layer_number}.self_attn.q_proj.bias",
                "model.layers.{layer_number}.self_attn.k_proj.bias",
                "model.layers.{layer_number}.self_attn.v_proj.bias",
            ],
        }
    )

    # 【修正点1】：完全移除 "model." 前缀，匹配真实 Key
    _MTP_MAPPING = {
        "enorm.weight": ["mtp.layers.{mtp_layer_index}.enorm.weight"],
        "hnorm.weight": ["mtp.layers.{mtp_layer_index}.hnorm.weight"],
        "eh_proj.weight": ["mtp.layers.{mtp_layer_index}.eh_proj.weight"],
        "final_layernorm.weight": ["mtp.layers.{mtp_layer_index}.final_layernorm.weight"],
    }

    def _weight_name_mapping_mtp(self, name: str, num_layers: int) -> list[str]:
        # 从 name 中提取 index，例如 mtp.layers.0 -> 0
        mtp_match = re.search(r"mtp\.layers\.(\d+)\.", name)
        mtp_layer_index = int(mtp_match.group(1)) if mtp_match else 0
        
        convert_names = []

        # 1. 处理特殊层 (enorm, hnorm 等)
        for keyword, mapping_names in self._MTP_MAPPING.items():
            if keyword in name:
                convert_names.extend([x.format(mtp_layer_index=mtp_layer_index) for x in mapping_names])
                return convert_names

        # 2. 处理 Sub-modules (MLP / Self Attention)
        # 真实 Key 结构: "mtp.layers.0.transformer_layer.mlp..."
        if "mlp" in name or "self_attention" in name:
            # 构造临时名字欺骗基类，获取标准 HF 命名 (model.layers.0.mlp...)
            suffix = name.split("transformer_layer.")[-1]
            temp_name = f"decoder.layers.{mtp_layer_index}.{suffix}"

            if "mlp" in name:
                hf_names = self._weight_name_mapping_mlp(temp_name)
            else:
                hf_names = self._weight_name_mapping_attention(temp_name)

            # 【修正点2】：精确替换路径
            # 将基类生成的 "model.layers.0" 替换为 "mtp.layers.0.transformer_layer"
            source_prefix = f"model.layers.{mtp_layer_index}"
            target_prefix = f"mtp.layers.{mtp_layer_index}.transformer_layer"
            
            convert_names = [x.replace(source_prefix, target_prefix) for x in hf_names]
            return convert_names

        raise NotImplementedError(f"Unsupported MTP parameter name: {name}")

    def _detect_layer_type_from_hf_model(self, layer_idx: int) -> str:
        """
        从 HF 模型的实际参数中检测指定层的类型。
        如果无法检测，返回 None。
        """
        if hasattr(self, 'safetensor_io') and hasattr(self.safetensor_io, 'index'):
            # 检查是否有 linear_attn 参数
            has_linear_attn = any(f"model.layers.{layer_idx}.linear_attn." in k for k in self.safetensor_io.index.keys())
            if has_linear_attn:
                return "linear_attention"
            else:
                # 检查是否有 self_attn 参数（排除 linear_attn 的情况）
                has_self_attn = any(f"model.layers.{layer_idx}.self_attn." in k for k in self.safetensor_io.index.keys())
                if has_self_attn:
                    return "full_attention"
        return None

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        # 检查 mcore_weights_name 是否是字符串
        if not isinstance(mcore_weights_name, str):
            # 如果不是字符串，可能是其他类型（如 Float16Module），直接调用基类方法
            return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        
        assert "_extra_state" not in mcore_weights_name
        
        # 清理 DDP 前缀，确保匹配
        clean_name = mcore_weights_name.replace("module.module.", "").replace("module.", "")

        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        
        # 优先检查直接映射
        if clean_name in direct_name_mapping:
            return [direct_name_mapping[clean_name]]
        
        # 路由
        if "mtp" in mcore_weights_name:
            return self._weight_name_mapping_mtp(mcore_weights_name, self.hf_config.num_hidden_layers)
        elif "self_attention" in mcore_weights_name:
            try:
                hf_names = self._weight_name_mapping_attention(mcore_weights_name)
                # 检查是否返回了空列表（应该被跳过的参数）
                if not hf_names:
                    # 对于应该被跳过的参数，返回一个占位符参数名
                    # bridge 基类会尝试加载它，如果不存在会报 KeyError
                    # 但我们可以通过重写 load_weights 来处理这种情况
                    # 或者返回一个存在的参数名作为占位符，然后在 _weight_to_mcore_format 中处理
                    # 为了简单，我们返回一个不存在的参数名，让 bridge 基类处理 KeyError
                    # 但 bridge 基类可能会在访问 hf_names[0] 时报错，所以我们需要重写 load_weights
                    # 暂时返回空列表，让 bridge 基类报错，然后我们重写 load_weights 来处理
                    return []
                return hf_names
            except NotImplementedError as e:
                # 如果参数应该被跳过（例如 linear_attention 层的 infllmv2_attn 参数），
                # 返回空列表，让 bridge 基类处理
                error_msg = str(e)
                if "linear_attention" in error_msg and "infllmv2_attn" in error_msg:
                    # 这是一个应该被跳过的参数，返回空列表
                    return []
                raise
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")
        
    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> tuple[list[str], list[torch.Tensor]]:
        # InfLLM V2 使用 query_key_value 格式，需要将 HF 的 q_proj/k_proj/v_proj 合并
        if ("self_attention.query_key_value." in mcore_weights_name or 
            "self_attention.infllmv2_attn.query_key_value." in mcore_weights_name) and "layer_norm" not in mcore_weights_name:
            # merge qkv for InfLLM V2 Attention Layers
            assert len(hf_weights) == 3
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            head_dim = getattr(self.hf_config, "head_dim", hidden_dim // num_attention_heads)
            q, k, v = hf_weights
            
            # Check if bias
            if ".bias" in mcore_weights_name:
                # InfLLM V2 格式：Q 在前，然后是 K，最后是 V
                # Q: (num_attention_heads * head_dim,)
                # K: (num_key_value_heads * head_dim,)
                # V: (num_key_value_heads * head_dim,)
                qkv = torch.cat([q, k, v], dim=0).contiguous()
                return qkv

            # Weight: 需要处理 TP 切分的情况
            # Q: (num_attention_heads * head_dim, hidden_size)
            # K: (num_key_value_heads * head_dim, hidden_size)
            # V: (num_key_value_heads * head_dim, hidden_size)
            # InfLLM V2 格式：直接拼接 Q, K, V（按照 Q -> K -> V 的顺序）
            qkv = torch.cat([q, k, v], dim=0).contiguous()
            return qkv

        # 处理 Full Attention layers (linear_qgkv 格式)
        if "self_attention.linear_qgkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            # merge qkv for Full Attention Layers
            assert len(hf_weights) == 3
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            num_querys_per_group = num_attention_heads // self.hf_config.num_key_value_heads
            head_dim = getattr(self.hf_config, "head_dim", hidden_dim // num_attention_heads)
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            
            # Check if bias
            if ".bias" in mcore_weights_name:
                 # Simple concat for bias
                 qgkv = torch.cat([q, k, v], dim=0).contiguous()
                 return qgkv

            # q k v might be tp split (Weight)
            real_num_key_value_heads = q.shape[0] // (2 * group_dim)
            q = (
                q.view(
                    [
                        real_num_key_value_heads,
                        num_querys_per_group,
                        2,
                        head_dim,
                        -1,
                    ]
                )
                .transpose(1, 2)
                .flatten(1, 3)
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim]

            qgkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qgkv

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        match = re.match(r"decoder\.layers\.(\d+)\.(self_attention\..+)", name)
        if match:
            layer_idx, rest = match.groups()
            layer_idx_int = int(layer_idx)
            hf_prefix = f"model.layers.{layer_idx}."
            
            # 获取该层的类型（linear_attention 或 full_attention）
            # 优先从 HF 模型的实际参数中检测，如果检测不到，再根据配置推断
            if not hasattr(self.hf_config, 'layer_types') or self.hf_config.layer_types is None:
                # 尝试从 HF 模型的实际参数中检测 layer_types
                layer_types = []
                num_hidden_layers = getattr(self.hf_config, 'num_hidden_layers', 28)
                
                # 如果 safetensor_io 已经初始化，可以从 index 中检测
                if hasattr(self, 'safetensor_io') and hasattr(self.safetensor_io, 'index'):
                    for i in range(num_hidden_layers):
                        detected_type = self._detect_layer_type_from_hf_model(i)
                        if detected_type:
                            layer_types.append(detected_type)
                        else:
                            # 如果检测不到，使用推断逻辑
                            linear_attention_interval = getattr(self.hf_config, 'linear_attention_interval', 4)
                            if i % linear_attention_interval == 0:
                                layer_types.append("linear_attention")
                            else:
                                layer_types.append("full_attention")
                else:
                    # 如果无法检测，使用推断逻辑
                    # 注意：根据实际 HF 模型，模式可能是每 4 层中，前 3 层是 linear_attention，第 4 层是 full_attention
                    # 但为了兼容性，先使用简单的推断逻辑：每 4 层的第 1 层是 linear_attention
                    linear_attention_interval = getattr(self.hf_config, 'linear_attention_interval', 4)
                    for i in range(num_hidden_layers):
                        if i % linear_attention_interval == 0:
                            layer_types.append("linear_attention")
                        else:
                            layer_types.append("full_attention")
                
                self.hf_config.layer_types = layer_types
            
            # 尝试从 HF 模型动态检测该层的类型（如果 safetensor_io 已初始化）
            detected_type = self._detect_layer_type_from_hf_model(layer_idx_int)
            if detected_type:
                layer_type = detected_type
            else:
                layer_type = self.hf_config.layer_types[layer_idx_int] if layer_idx_int < len(self.hf_config.layer_types) else "full_attention"
            
            # 调试信息：打印前几层的权重名称
            if layer_idx_int < 3:
                print(f"[DEBUG Bridge] Layer {layer_idx}: layer_type={layer_type}, mapping '{name}' -> rest='{rest}'")
            
            # 处理 InfLLM V2 attention 格式
            # 注意：只有当 layer_type 是 full_attention 时，才应该处理 infllmv2_attn 参数
            if rest.startswith("self_attention.infllmv2_attn."):
                # 如果该层是 linear_attention，不应该有 infllmv2_attn 参数
                # 这可能表示模型配置不一致，但为了兼容性，我们仍然尝试映射
                if layer_type == "linear_attention":
                    # 对于 linear_attention 层，infllmv2_attn 参数不应该存在
                    # 但 Megatron 模型可能期望这些参数
                    # 
                    # 问题的本质：Megatron 模型构建时所有层都被识别为 full_attention，
                    # 但实际上 HF 模型中只有部分层是 full_attention
                    # 
                    # 解决方案：直接抛出 NotImplementedError，让 bridge 基类处理
                    # bridge 基类会捕获这个异常并跳过该参数
                    raise NotImplementedError(
                        f"Layer {layer_idx} is linear_attention but Megatron model expects infllmv2_attn parameters. "
                        f"This parameter should be skipped as it doesn't exist in the HF model."
                    )
                
                sub_name = rest[len("self_attention.infllmv2_attn.") :]
                if sub_name == "query_key_value.weight":
                    return [
                        hf_prefix + "self_attn.q_proj.weight",
                        hf_prefix + "self_attn.k_proj.weight",
                        hf_prefix + "self_attn.v_proj.weight",
                    ]
                elif sub_name == "query_key_value.bias":
                    return [
                        hf_prefix + "self_attn.q_proj.bias",
                        hf_prefix + "self_attn.k_proj.bias",
                        hf_prefix + "self_attn.v_proj.bias",
                    ]
                elif sub_name == "dense.weight":
                    return [hf_prefix + "self_attn.o_proj.weight"]
                elif sub_name == "dense.bias":
                    return [hf_prefix + "self_attn.o_proj.bias"]
                elif sub_name == "q_layernorm.weight":
                    return [hf_prefix + "self_attn.q_norm.weight"]
                elif sub_name == "k_layernorm.weight":
                    return [hf_prefix + "self_attn.k_norm.weight"]
                else:
                    # 其他参数直接映射
                    return [hf_prefix + sub_name.replace("infllmv2_attn.", "self_attn.")]
            
            # 处理标准的 query_key_value 格式
            # 注意：需要根据 layer_type 判断是 linear attention 还是 full attention
            if rest.startswith("self_attention.query_key_value."):
                sub_name = rest[len("self_attention.query_key_value.") :]
                # 如果是 linear attention 层，query_key_value 应该映射到 linear attention 的权重
                if layer_type == "linear_attention":
                    # Linear attention 不使用 query_key_value，应该使用其他权重名称
                    # 但实际上，如果 Megatron 模型中有 query_key_value，可能是错误的
                    # 这里先尝试映射到 linear attention 的权重
                    # 注意：linear attention 的权重名称是 A_log, in_proj_qkvz 等，不是 q_proj
                    # 如果遇到这种情况，可能需要检查模型架构构建是否正确
                    print(f"[WARNING] Layer {layer_idx} is linear_attention but has query_key_value weight. "
                          f"This might indicate a model architecture mismatch.")
                    # 暂时返回空列表，让基类处理
                    return []
                else:
                    # Full attention 层，正常映射到 q_proj/k_proj/v_proj
                    if sub_name == "weight":
                        return [
                            hf_prefix + "self_attn.q_proj.weight",
                            hf_prefix + "self_attn.k_proj.weight",
                            hf_prefix + "self_attn.v_proj.weight",
                        ]
                    elif sub_name == "bias":
                        return [
                            hf_prefix + "self_attn.q_proj.bias",
                            hf_prefix + "self_attn.k_proj.bias",
                            hf_prefix + "self_attn.v_proj.bias",
                        ]
            
            # 处理 linear_attn 格式
            # 注意：根据 Megatron 转 HF 的脚本，linear_attn 参数映射到 layer.linear_attn.xxx
            # 而不是 layer.self_attn.linear_attn.xxx（在HF模型中，linear_attn是直接作为layer的一个属性）
            if rest.startswith("self_attention.linear_attn."):
                sub_name = rest[len("self_attention.linear_attn.") :]
                # HF 模型中使用 linear_attn.* 直接作为 layer 的属性
                return [hf_prefix + "linear_attn." + sub_name]
            
            # 处理 self_attn 格式（用于直接映射的情况）
            if rest.startswith("self_attention.self_attn."):
                sub_name = rest[len("self_attention.self_attn.") :]
                return [hf_prefix + "self_attn." + sub_name]
            
            # 处理 input_layernorm（Qwen3 Next 架构中，input_layernorm 在 Attention 模块内）
            # 但在 HF 模型中，它直接作为 layer 的属性
            if rest == "self_attention.input_layernorm.weight":
                return [hf_prefix + "input_layernorm.weight"]
        
        return super()._weight_name_mapping_attention(name)

    def _build_config(self):
        """
        构建 Megatron 配置。
        
        注意：InfLLM V2 的特殊参数（infllmv2_topk_blocks, infllmv2_block_size, infllmv2_use_stage1）
        应该通过命令行参数传递，而不是在这里设置，因为它们不是 TransformerConfig 的一部分。
        """
        # 1. 先构建基础配置 (包含 hidden_size, num_layers 等通用参数)
        config = self._build_base_config(
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_experts,
            moe_aux_loss_coeff=self.hf_config.router_aux_loss_coef,
            moe_router_load_balancing_type="none",
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Qwen specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,  # Qwen3 Next 使用 QK LayerNorm
            # Qwen3 Next specific
            use_gated_attention=True,  # Qwen3 Next 使用 Gated Delta Net
        )

        return config

    def load_weights(self, model, hf_model_path: str, memory_efficient: bool = True):
        """
        重写 load_weights 方法，处理应该被跳过的参数（返回空列表的情况）。
        
        问题的本质：
        - Megatron 模型构建时所有层都被识别为 full_attention（有 infllmv2_attn 参数）
        - 但 HF 模型中只有部分层是 full_attention，其他层是 linear_attention（只有 linear_attn 参数）
        - 当 linear_attention 层遇到 infllmv2_attn 参数时，应该跳过这些参数
        
        注意：model 可能是单个模型对象，也可能是列表（pipeline parallelism 的情况）
        """
        from mbridge.core.safetensor_io import SafeTensorIO
        
        # 初始化 safetensor_io（如果还没有初始化）
        if not hasattr(self, 'safetensor_io') or self.safetensor_io is None:
            self.safetensor_io = SafeTensorIO(hf_model_path)
        
        # 处理 model 是列表的情况（pipeline parallelism）
        if isinstance(model, (list, tuple)):
            models_to_process = model
        else:
            models_to_process = [model]
        
        # 过滤掉应该被跳过的参数，并加载其他参数
        skipped_params = []
        loaded_params = []
        
        # 遍历所有模型（在 pipeline parallelism 的情况下）
        for m in models_to_process:
            # 获取所有需要加载的参数
            mcore_param_names = list(m.named_parameters())
            
            for name, param in mcore_param_names:
                # 获取对应的 HF 参数名
                try:
                    hf_names = self._weight_name_mapping_mcore_to_hf(name)
                    if not hf_names:
                        # 如果返回空列表，说明这个参数应该被跳过
                        skipped_params.append(name)
                        continue
                    
                    # 加载权重
                    try:
                        hf_weights = [self.safetensor_io.load_one_hf_weight(hf_name) for hf_name in hf_names]
                        mcore_weight = self._weight_to_mcore_format(name, hf_weights)
                        param.data.copy_(mcore_weight)
                        loaded_params.append(name)
                    except KeyError as e:
                        # 如果参数不存在，跳过
                        skipped_params.append(name)
                        continue
                except NotImplementedError as e:
                    # 如果抛出 NotImplementedError，也跳过这个参数
                    error_msg = str(e)
                    if "linear_attention" in error_msg and "infllmv2_attn" in error_msg:
                        skipped_params.append(name)
                        continue
                    raise
        
        if skipped_params:
            print(f"[INFO] Skipped {len(skipped_params)} parameters that don't exist in HF model:")
            for name in skipped_params[:10]:  # 只打印前10个
                print(f"  - {name}")
            if len(skipped_params) > 10:
                print(f"  ... and {len(skipped_params) - 10} more")
        
        print(f"[INFO] Loaded {len(loaded_params)} parameters from HF model")

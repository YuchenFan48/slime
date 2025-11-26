import re
import torch
from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


@register_model("qwen3_kimi")
class Qwen3KimiBridge(Qwen2MoEBridge):
    _ATTENTION_MAPPING = (
        Qwen2MoEBridge._ATTENTION_MAPPING
        | {
            f"self_attention.{weight_name}": ["model.layers.{layer_number}." + weight_name]
            for weight_name in [
                "input_layernorm.weight",
                # === Kimi Linear Attention Components ===
                # Parameters
                "linear_attn.A_log",
                "linear_attn.dt_bias",
                # Norms
                "linear_attn.o_norm.weight",
                # Convolutions (Separated)
                "linear_attn.q_conv1d.weight",
                "linear_attn.k_conv1d.weight",
                "linear_attn.v_conv1d.weight",
                # Projections (Separated)
                "linear_attn.q_proj.weight",
                "linear_attn.k_proj.weight",
                "linear_attn.v_proj.weight",
                "linear_attn.f_a_proj.weight",
                "linear_attn.f_b_proj.weight",
                "linear_attn.g_a_proj.weight",
                "linear_attn.g_b_proj.weight",
                "linear_attn.b_proj.weight",
                "linear_attn.o_proj.weight",
                # === Standard / Gated Attention Components (Full Attn Layers) ===
                "self_attn.k_norm.weight",
                "self_attn.q_norm.weight",
                "self_attn.o_proj.weight",
            ]
        }
        | {
            # Standard Qwen Full Attention Layer Mappings (Merged QKV)
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

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
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
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")
        
    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> tuple[list[str], list[torch.Tensor]]:
        # This logic handles the Full Attention layers (which usually use linear_qgkv naming in Megatron)
        # Kimi's Linear Attention layers use specific names (linear_attn.*) and won't trigger this.
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
            hf_prefix = f"model.layers.{layer_idx}."
            if rest.startswith("self_attention.linear_attn."):
                sub_name = rest[len("self_attention.linear_attn.") :]
                return [hf_prefix + "linear_attn." + sub_name]
            if rest.startswith("self_attention.self_attn."):
                sub_name = rest[len("self_attention.self_attn.") :]
                return [hf_prefix + "self_attn." + sub_name]
        return super()._weight_name_mapping_attention(name)

    def _build_config(self):
            # 1. 先构建基础配置 (包含 hidden_size, num_layers 等通用参数)
            config = self._build_base_config(
                use_cpu_initialization=False,
                moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
                moe_router_bias_update_rate=0.001,
                moe_router_topk=self.hf_config.num_experts_per_tok,
                num_moe_experts=self.hf_config.num_experts,
                moe_aux_loss_coeff=self.hf_config.router_aux_loss_coef,
                moe_router_load_balancing_type="none",
                moe_grouped_gemm=True,
                moe_router_score_function="softmax",
                persist_layer_norm=True,
                bias_activation_fusion=True,
                bias_dropout_fusion=True,
                moe_router_pre_softmax=False,
                qk_layernorm=True,
                use_gated_attention=False, # Kimi 使用 KDA，不使用 Next 的 Gated Delta
            )

            # 2. 【关键修复】显式注入 Kimi 特有的参数
            # 如果不加这几行，Mcore 模型就会用错误的默认值初始化，导致 expected=8192
            config.linear_num_value_heads = getattr(self.hf_config, "linear_num_value_heads", 16)
            config.linear_num_key_heads = getattr(self.hf_config, "linear_num_key_heads", 16)
            config.linear_value_head_dim = getattr(self.hf_config, "linear_value_head_dim", 128)
            config.linear_key_head_dim = getattr(self.hf_config, "linear_key_head_dim", 128)
            config.linear_conv_kernel_dim = getattr(self.hf_config, "linear_conv_kernel_dim", 4)

            return config

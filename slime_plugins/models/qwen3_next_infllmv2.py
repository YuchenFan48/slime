"""
Qwen3-Next model with InfLLM V2 sparse attention integration.

注意：Qwen3-Next 使用 linear attention (Gated Delta Net)，而 InfLLM V2 是为标准 attention 设计的。
本实现提供了一个混合方案：对于 full_attention 层使用 InfLLM V2，对于 linear_attention 层保持原有实现。
"""

import copy
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig

from .hf_attention import HuggingfaceAttention
from .infllmv2_attention import InfLLMV2Attention

# 导入原 qwen3_next 模块中的组件
try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm
except ImportError:
    Qwen3NextRMSNorm = None

# 导入 Qwen3NextGatedDeltaNet（定义在 qwen3_next.py 中）
try:
    from .qwen3_next import Qwen3NextGatedDeltaNet, Attention as Qwen3NextAttention
except ImportError:
    Qwen3NextGatedDeltaNet = None
    Qwen3NextAttention = None


class Qwen3NextInfLLMV2Attention(HuggingfaceAttention):
    """
    Qwen3-Next Attention with InfLLM V2 support.
    
    对于 full_attention 层，使用 InfLLM V2 稀疏 attention。
    对于 linear_attention 层，保持原有的 Qwen3NextGatedDeltaNet 实现。
    """
    
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        model_comm_pgs=None,
        use_infllmv2: bool = True,
        attn_mask_type=None,  # MTP requires this parameter to be present
    ):
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            model_comm_pgs,
        )
        
        self.use_infllmv2 = use_infllmv2
        
        # CRITICAL: Qwen3-Next architecture requires input_layernorm before attention
        # This must be created for BOTH InfLLM V2 and linear attention paths
        # This matches the standard qwen3_next.py implementation
        if Qwen3NextRMSNorm is None:
            raise ImportError(
                "Qwen3NextRMSNorm is not available. Please install transformers>=4.35.0:\n"
                "pip install transformers>=4.35.0"
            )
        self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)        

        if use_infllmv2:
            # 使用 InfLLM V2 attention
            num_attention_heads = args.num_attention_heads
            num_query_groups = args.num_query_groups if hasattr(args, 'num_query_groups') else num_attention_heads
            hidden_size = args.hidden_size
            attention_dropout = args.attention_dropout if hasattr(args, 'attention_dropout') else 0.0
            
            # InfLLM V2 参数
            topk_blocks = getattr(args, 'infllmv2_topk_blocks', 64)
            block_size = getattr(args, 'infllmv2_block_size', 64)
            use_stage1 = getattr(args, 'infllmv2_use_stage1', True)
            # QK LayerNorm for numerical stability (Qwen3-Next architecture)
            qk_layernorm = getattr(args, 'qk_layernorm', False)
            
            self.infllmv2_attn = InfLLMV2Attention(
                config=config,
                layer_number=layer_number,
                num_attention_heads=num_attention_heads,
                num_query_groups=num_query_groups,
                hidden_size=hidden_size,
                attention_dropout=attention_dropout,
                topk_blocks=topk_blocks,
                block_size=block_size,
                use_stage1=use_stage1,
                qk_layernorm=qk_layernorm,
            )
        else:
            # 使用原有的 Qwen3NextGatedDeltaNet（linear attention）
            if Qwen3NextGatedDeltaNet is None:
                raise ImportError(
                    "Qwen3NextGatedDeltaNet is not available. Please ensure qwen3_next module is available. "
                    "Also install fla: pip install fla-attn"
                )
            
            self.linear_attn = Qwen3NextGatedDeltaNet(self.hf_config, self.hf_layer_idx)
            # self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)
    
    def hf_forward(self, hidden_states, position_ids, packed_seq_params):
        # CRITICAL: Qwen3-Next architecture requires input_layernorm before attention
        # This matches the standard qwen3_next.py implementation
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_infllmv2:
            # 使用 InfLLM V2 attention
            # hidden_states 格式: (batch_size, seq_len, hidden_size)
            # InfLLMV2Attention 需要 (seq_len, batch_size, hidden_size) 格式
            hidden_states = hidden_states.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
            
            # 调用 InfLLM V2 attention
            output, _ = self.infllmv2_attn(
                hidden_states,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
            )
            
            # 转换回 (batch_size, seq_len, hidden_size)
            output = output.transpose(0, 1)
            return output
        else:
            # 使用原有的 linear attention
            # Note: input_layernorm is already applied above for both paths
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cu_seqlens=packed_seq_params.cu_seqlens_q,
            )
            return hidden_states


def get_qwen3_next_infllmv2_spec(args, config, vp_stage):
    """
    Get Qwen3-Next model specification with InfLLM V2 sparse attention.
    
    策略：
    - 对于 full_attention 层：使用 InfLLM V2 稀疏 attention
    - 对于 linear_attention 层：保持原有的 Qwen3NextGatedDeltaNet 实现
    """
    # always use the moe path
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    for layer_id in range(num_layers_to_build):
        layer_type = hf_config.layer_types[layer_id + offset]
        
        if layer_type == "full_attention":
            # 对于 full_attention 层，使用 InfLLM V2
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Qwen3NextInfLLMV2Attention,
                params={
                    "args": args,
                    "use_infllmv2": True,
                    "attn_mask_type": AttnMaskType.causal,  # MTP requires attn_mask_type to be set
                },
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
        elif layer_type == "linear_attention":
            # 对于 linear_attention 层，可以选择使用 InfLLM V2 或保持原有实现
            # 注意：InfLLM V2 是为标准 attention 设计的，可能不完全兼容 linear attention
            # 这里提供一个选项，默认保持原有实现
            use_infllmv2_for_linear = getattr(args, 'infllmv2_use_for_linear_attention', False)
            
            if use_infllmv2_for_linear:
                layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
                layer_specs.submodules.self_attention = ModuleSpec(
                    module=Qwen3NextInfLLMV2Attention,
                    params={
                        "args": args,
                        "use_infllmv2": True,
                        "attn_mask_type": AttnMaskType.causal,  # MTP requires attn_mask_type to be set
                    },
                )
                transformer_layer_spec.layer_specs[layer_id] = layer_specs
            else:
                # 保持原有的 linear attention 实现（从 qwen3_next.py 导入）
                if Qwen3NextAttention is None:
                    raise ImportError(
                        "Cannot import Qwen3NextAttention from qwen3_next module. "
                        "Please ensure qwen3_next.py is available and all dependencies are installed."
                    )
                
                layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
                layer_specs.submodules.self_attention = ModuleSpec(
                    module=Qwen3NextAttention,
                    params={
                        "args": args,
                        "attn_mask_type": AttnMaskType.causal,  # MTP requires attn_mask_type to be set
                    },
                )
                transformer_layer_spec.layer_specs[layer_id] = layer_specs
        
        transformer_layer_spec.layer_specs[layer_id].submodules.mlp.submodules.shared_experts.params = {"gate": True}
    
    return transformer_layer_spec


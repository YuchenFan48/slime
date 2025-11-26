"""
GPT-OSS model with InfLLM V2 sparse attention integration.

This module extends GPT-OSS to use InfLLM V2's two-stage sparse attention.
"""

######################################## Monkey Patch ########################################
# Add local deepmega path
import sys
deepmega_path = "/mnt/shared-storage-user/p1-shared/yuchenzhang"
if deepmega_path not in sys.path:
    sys.path.insert(0, deepmega_path)

# Apply deepmega patches to support MoE + bias
from deepmega.inplace_megatron import mock_megatron
mock_megatron()

from deepmega.models.llm.gpt_oss.model_patch import core_transformer_config_from_args
import megatron.training.arguments as arguments
arguments.core_transformer_config_from_args = core_transformer_config_from_args
######################################## Monkey Patch ########################################

import copy
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.spec_utils import ModuleSpec

from .infllmv2_attention import InfLLMV2Attention


def get_gpt_oss_infllmv2_spec(args, config, vp_stage):
    """
    Get GPT-OSS model specification with InfLLM V2 sparse attention.
    
    Args:
        args: Training arguments
        config: Transformer config
        vp_stage: Virtual pipeline stage
        
    Returns:
        Transformer layer spec with InfLLM V2 attention
    """
    # Always use the MoE path for GPT-OSS
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers
    
    # Define the decoder block spec with TE
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
    
    # Replace attention module with InfLLM V2 attention
    for layer_id in range(num_layers_to_build):
        layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
        
        # Get attention configuration from args
        num_attention_heads = args.num_attention_heads
        num_query_groups = args.num_query_groups if hasattr(args, 'num_query_groups') else num_attention_heads
        hidden_size = args.hidden_size
        attention_dropout = args.attention_dropout if hasattr(args, 'attention_dropout') else 0.0
        
        # InfLLM V2 specific parameters
        topk_blocks = getattr(args, 'infllmv2_topk_blocks', 64)
        block_size = getattr(args, 'infllmv2_block_size', 64)
        use_stage1 = getattr(args, 'infllmv2_use_stage1', True)
        
        # Replace self_attention module
        layer_specs.submodules.self_attention = ModuleSpec(
            module=InfLLMV2Attention,
            params={
                "num_attention_heads": num_attention_heads,
                "num_query_groups": num_query_groups,
                "hidden_size": hidden_size,
                "attention_dropout": attention_dropout,
                "topk_blocks": topk_blocks,
                "block_size": block_size,
                "use_stage1": use_stage1,
                "attn_mask_type": AttnMaskType.causal,  # MTP requires attn_mask_type to be set
            },
        )
        
        transformer_layer_spec.layer_specs[layer_id] = layer_specs
    
    return transformer_layer_spec




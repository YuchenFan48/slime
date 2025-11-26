"""
GPT-OSS model specification for Megatron training.

This module provides model spec for GPT-OSS MoE models with bias support.
Requires deepmega to bypass Megatron's MoE + bias limitation.
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
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def get_gpt_oss_spec(args, config, vp_stage):
    """
    Get GPT-OSS model specification.
    
    GPT-OSS is a MoE model with bias in all linear layers.
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
    
    # GPT-OSS uses standard MoE architecture with bias
    # The bias configuration is handled by config.add_bias_linear
    
    return transformer_layer_spec


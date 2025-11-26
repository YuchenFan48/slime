import copy
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig
from transformers.activations import ACT2FN
from typing import Any, List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
import sys
import os
fla_path = '/apdcephfs/mnt/cephfs/users/yuchenfan/flash-linear-attention'

# 只在路径存在且尚未在 sys.path 中时才添加
if os.path.exists(fla_path) and fla_path not in sys.path:
    sys.path.insert(0, fla_path)  # insert at front to prioritize

from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextRMSNorm,
    Qwen3NextRotaryEmbedding,
)

from .hf_attention import HuggingfaceAttention

    
from transformers import AutoConfig, Qwen2Config

# === 新增代码开始 ===
class Qwen3KimiConfig(Qwen2Config):
    model_type = "qwen3_kimi"

    def __init__(
        self,
        linear_conv_kernel_dim=4,
        linear_num_value_heads=None,
        linear_num_key_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        attention_bias=False,  # <--- 新增默认值，Qwen 系列通常 QKV 有 bias
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.attention_bias = attention_bias  # <--- 新增赋值

# 注册配置，确保 AutoConfig 能识别
try:
    AutoConfig.register("qwen3_kimi", Qwen3KimiConfig)
except ValueError:
    pass
# === 新增代码结束 ===

def check_for_nan(tensor, name):
    """
    Checks a tensor for NaN values and raises a ValueError if any are found.
    """
    if torch.isnan(tensor).any():
        # Create a detailed error message
        error_message = (
            f"NaN detected in tensor: '{name}'\n"
            f"Tensor shape: {tensor.shape}\n"
            f"Tensor device: {tensor.device}\n"
            f"Tensor dtype: {tensor.dtype}\n"
            # Printing the whole tensor can be too large, so let's find where the NaNs are
            f"NaN locations (indices): {torch.isnan(tensor).nonzero(as_tuple=True)}"
        )
        # Raise the error with the detailed message
        raise ValueError(error_message)

class KimiDeltaAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_conv_kernel_dim
        self.head_dim = config.linear_value_head_dim
        self.num_heads = config.linear_num_value_heads
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.layer_idx = layer_idx

        assert self.mode in [
            'chunk', 'fused_recurrent'], f"Not suppoerted mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(
            self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu',
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation='silu'
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation='silu'
        )

        self.A_log = torch.nn.Parameter(torch.log(torch.empty(
            self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1))

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=torch.float32))
        nn.init.uniform_(self.dt_bias, a=0.001, b=0.01)

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        cache_params=None,
        **kwargs
    ):
        # 检查输入
        if check_for_nan(hidden_states, "input hidden_states"):
            return hidden_states  # 返回原值以便调试
        
        use_cache = cache_params is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get('cu_seqlens', None)
        indices = None

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        if cache_params is not None:
            if cache_params.conv_states[self.layer_idx] is not None:
                conv_state_q, conv_state_k, conv_state_v = cache_params.conv_states[
                    self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        
        # 检查投影层输出
        q_proj = self.q_proj(hidden_states)
        if check_for_nan(q_proj, "q_proj"):
            return q_proj
        
        k_proj = self.k_proj(hidden_states)
        if check_for_nan(k_proj, "k_proj"):
            return k_proj
            
        v_proj = self.v_proj(hidden_states)
        if check_for_nan(v_proj, "v_proj"):
            return v_proj
        
        # 检查卷积层输出
        q, conv_state_q = self.q_conv1d(
            x=q_proj,
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        if check_for_nan(q, "q after conv1d"):
            return q
        
        k, conv_state_k = self.k_conv1d(
            x=k_proj,
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        if check_for_nan(k, "k after conv1d"):
            return k
        
        v, conv_state_v = self.v_conv1d(
            x=v_proj,
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens
        )
        if check_for_nan(v, "v after conv1d"):
            return v
        
        # 检查门控机制
        g_a = self.f_a_proj(hidden_states)
        if check_for_nan(g_a, "f_a_proj"):
            return g_a
        
        g = self.f_b_proj(g_a)
        if check_for_nan(g, "f_b_proj"):
            return g
        
        # 这里是最可能出现 NaN 的地方
        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
        if check_for_nan(g, "g after fused_kda_gate"):
            return g
        
        beta = self.b_proj(hidden_states).float().sigmoid()
        if check_for_nan(beta, "beta"):
            return beta

        q, k = map(lambda x: rearrange(
            x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if mode == 'chunk':
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            if check_for_nan(o, "output from chunk_kda"):
                return o
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            if check_for_nan(o, "output from fused_recurrent_kda"):
                return o
                
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = recurrent_state
            cache_params.conv_states[self.layer_idx] = (
                conv_state_q, conv_state_k, conv_state_v)

        g = self.g_b_proj(self.g_a_proj(hidden_states))
        if check_for_nan(g, "g after g_b_proj"):
            return g
        
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
        o = self.o_norm(o, g)
        if check_for_nan(o, "o after o_norm"):
            return o

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if check_for_nan(o, "final output"):
            return o
            
        return o

class Attention(HuggingfaceAttention):
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        model_comm_pgs=None,
    ):
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            model_comm_pgs,
        )
        if Qwen3NextAttention is None:
            raise ImportError("Please install transformers>=4.35.0 to use Qwen3NextAttention.")

        self.layer_type = self.hf_config.layer_types[self.hf_layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = KimiDeltaAttention(self.hf_config, self.hf_layer_idx)
        elif self.layer_type == "full_attention":
            self.rotary_emb = Qwen3NextRotaryEmbedding(config=self.hf_config)
            self.self_attn = Qwen3NextAttention(self.hf_config, self.hf_layer_idx)

        self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    def hf_forward(self, hidden_states, position_ids, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cu_seqlens=packed_seq_params.cu_seqlens_q,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        return hidden_states


def get_qwen3_next_spec(args, config, vp_stage):
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

    for layer_id in range(num_layers_to_build):
        transformer_layer_spec.layer_specs[layer_id].submodules.self_attention = ModuleSpec(
            module=Attention,
            params={"args": args},
        )
        transformer_layer_spec.layer_specs[layer_id].submodules.mlp.submodules.shared_experts.params = {"gate": True}

    return transformer_layer_spec

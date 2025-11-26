"""
InfLLM V2 Sparse Attention integration for Megatron training.

This module provides a custom attention implementation using InfLLM V2's
two-stage sparse attention mechanism.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from megatron.core import mpu, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.packed_seq_params import PackedSeqParams

try:
    from infllm_v2 import (
        infllmv2_attn_varlen_func,
        infllmv2_attn_stage1,
    )
    INFLLMV2_AVAILABLE = True
except ImportError:
    INFLLMV2_AVAILABLE = False
    print("Warning: infllm_v2 not available. Please install infllmv2_cuda_impl.")

# Try to import Qwen3NextRMSNorm for QK LayerNorm
try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm
    QWEN3_RMSNORM_AVAILABLE = True
except ImportError:
    QWEN3_RMSNORM_AVAILABLE = False
    Qwen3NextRMSNorm = None


class InfLLMV2Attention(MegatronModule):
    """
    InfLLM V2 Sparse Attention module.
    
    This module implements two-stage sparse attention:
    1. Stage 1: Top-K context selection (computes relevance scores and selects top-k blocks)
    2. Stage 2: Sparse attention computation on selected blocks
    """
    
    def __init__(
        self,
        config,
        layer_number: int,
        num_attention_heads: int,
        num_query_groups: int,
        hidden_size: int,
        attention_dropout: float = 0.0,
        topk_blocks: int = 64,  # Number of blocks to select in Stage 1
        block_size: int = 64,   # Block size for sparse attention
        use_stage1: bool = True,  # Whether to use Stage 1 for top-k selection
        qk_layernorm: bool = False,  # Whether to apply QK LayerNorm
    ):
        super().__init__(config=config)
        
        if not INFLLMV2_AVAILABLE:
            raise ImportError(
                "infllm_v2 is not available. Please install infllmv2_cuda_impl:\n"
                "cd /mnt/shared-storage-user/p1-shared/yuchenzhang/infllmv2_cuda_impl && pip install -e ."
            )
        
        self.layer_number = layer_number
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.topk_blocks = topk_blocks
        self.block_size = block_size
        self.use_stage1 = use_stage1
        self.qk_layernorm = qk_layernorm
        
        # Get tensor parallel size
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        
        # Calculate per-partition head counts (for TP)
        self.num_attention_heads_per_partition = num_attention_heads // self.tp_size
        self.num_query_groups_per_partition = num_query_groups // self.tp_size
        self.hidden_size_per_partition = hidden_size // self.tp_size
        
        # Group size for GQA/MQA
        self.group_size = num_attention_heads // num_query_groups
        
        # QKV projection layers
        # For GQA: Q has num_attention_heads, K/V have num_query_groups
        qkv_size = (num_attention_heads + 2 * num_query_groups) * self.head_dim
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            qkv_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False,
        )
        
        # QK LayerNorm for numerical stability (Qwen3-Next architecture)
        # Note: input_layernorm is handled by the wrapper (e.g., Qwen3NextInfLLMV2Attention)
        # or by the transformer block (e.g., GPT-OSS), not here
        if self.qk_layernorm:
            if not QWEN3_RMSNORM_AVAILABLE:
                raise ImportError(
                    "Qwen3NextRMSNorm is not available. Please install transformers>=4.35.0:\n"
                    "pip install transformers>=4.35.0"
                )
            # Get norm_epsilon from config
            norm_epsilon = getattr(config, 'norm_epsilon', 1e-6)
            
            # Create Q and K LayerNorm layers
            # Each head gets its own normalization
            self.q_layernorm = Qwen3NextRMSNorm(self.head_dim, eps=norm_epsilon)
            self.k_layernorm = Qwen3NextRMSNorm(self.head_dim, eps=norm_epsilon)
        
        # Output projection
        self.dense = tensor_parallel.RowParallelLinear(
            hidden_size,
            hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
        )
    
    def _generate_topk_indices(
        self,
        scores: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """
        Generate top-k block indices from Stage 1 scores.
        
        Args:
            scores: Attention scores from Stage 1, shape (nheads_k, total_q, max_seqlen_k)
            cu_seqlens_q: Cumulative sequence lengths for queries
            max_seqlen_k: Maximum key sequence length
            
        Returns:
            topk_idx: Top-k block indices, shape (nheads_k, total_q, topk_blocks)
        """
        nheads_k, total_q, max_seqlen_k_actual = scores.shape
        # 使用实际序列长度计算 block 数量（而不是 max_seqlen_k 参数）
        num_blocks_k = (max_seqlen_k_actual + self.block_size - 1) // self.block_size
        
        # Pad scores to align with block size
        pad_length = num_blocks_k * self.block_size - max_seqlen_k_actual
        if pad_length > 0:
            # Pad with 0 (not -inf) since Stage1 returns probabilities [0, 1]
            # Padded positions should not contribute to block scores
            pad_tensor = torch.zeros(
                (nheads_k, total_q, pad_length),
                dtype=scores.dtype,
                device=scores.device
            )
            scores_blocks = torch.cat([scores, pad_tensor], dim=-1)
        else:
            scores_blocks = scores
        
        # Reshape scores to block-level: (nheads_k, total_q, num_blocks_k, block_size)
        # Average scores within each block
        scores_blocks = scores_blocks.reshape(nheads_k, total_q, num_blocks_k, self.block_size)
        
        # Handle Inf values: if any element in a block is Inf, set block score to 0
        # This prevents Inf from propagating through mean operation
        has_inf = scores_blocks.isinf().any(dim=-1, keepdim=True)  # (nheads_k, total_q, num_blocks_k, 1)
        scores_blocks_clean = torch.where(has_inf, torch.zeros_like(scores_blocks), scores_blocks)
        
        block_scores = scores_blocks_clean.mean(dim=-1)  # (nheads_k, total_q, num_blocks_k)
        
        # Select top-k blocks for each query
        topk_blocks = min(self.topk_blocks, num_blocks_k)
        _, topk_indices = torch.topk(block_scores, k=topk_blocks, dim=-1)  # (nheads_k, total_q, topk_blocks)
        
        # Convert to int32 format expected by infllmv2
        topk_idx = topk_indices.to(torch.int32)
        
        return topk_idx
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[dict] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with InfLLM V2 sparse attention.
        
        Args:
            hidden_states: Input hidden states, shape (seq_len, batch_size, hidden_size)
            attention_mask: Attention mask (not used in varlen format)
            key_value_states: Key-value states for cross-attention (not used)
            inference_context: Inference context (for KV cache)
            rotary_pos_emb: Rotary position embeddings
            rotary_pos_cos: Rotary position cosine embeddings
            rotary_pos_sin: Rotary position sine embeddings
            attention_bias: Attention bias (not used)
            packed_seq_params: Packed sequence parameters with cu_seqlens
            sequence_len_offset: Sequence length offset
            inference_params: Inference parameters (for KV cache)
            
        Returns:
            output: Output tensor, shape (seq_len, batch_size, hidden_size)
            attention_bias: Attention bias (None for this implementation)
        """
        if packed_seq_params is None:
            raise ValueError("packed_seq_params is required for InfLLM V2 attention")
        
        seq_len, batch_size, hidden_size = hidden_states.shape
        
        # QKV projection (returns output, bias tuple)
        # Note: with gather_output=False, output is sharded across TP ranks
        qkv, qkv_bias = self.query_key_value(hidden_states)
        if qkv_bias is not None:
            qkv = qkv + qkv_bias
        
        # For GQA, Q/K/V have different sizes, so we need to split manually
        # qkv shape: (seq_len, batch_size, (num_attention_heads_per_partition + 2 * num_query_groups_per_partition) * head_dim)
        q_dim = self.num_attention_heads_per_partition * self.head_dim
        kv_dim = self.num_query_groups_per_partition * self.head_dim
        
        # Split Q, K, V
        q = qkv[..., :q_dim]  # (seq_len, batch_size, q_dim)
        k = qkv[..., q_dim:q_dim+kv_dim]  # (seq_len, batch_size, kv_dim)
        v = qkv[..., q_dim+kv_dim:]  # (seq_len, batch_size, kv_dim)
        
        # Reshape to separate heads
        q = q.view(seq_len, batch_size, self.num_attention_heads_per_partition, self.head_dim)
        k = k.view(seq_len, batch_size, self.num_query_groups_per_partition, self.head_dim)
        v = v.view(seq_len, batch_size, self.num_query_groups_per_partition, self.head_dim)
        
        # Apply QK LayerNorm for numerical stability (Qwen3-Next architecture)
        # This prevents Q/K values from growing unbounded during training
        if self.qk_layernorm:
            # Apply LayerNorm to each head independently
            # Q: (seq_len, batch_size, num_attention_heads_per_partition, head_dim)
            # K: (seq_len, batch_size, num_query_groups_per_partition, head_dim)
            # We need to apply norm along the head_dim dimension for each head
            q = self.q_layernorm(q)  # RMSNorm normalizes along last dimension
            k = self.k_layernorm(k)
        
        # Get sequence length information
        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        cu_seqlens_k = packed_seq_params.cu_seqlens_k if hasattr(packed_seq_params, 'cu_seqlens_k') else cu_seqlens_q
        max_seqlen_q = packed_seq_params.max_seqlen_q
        max_seqlen_k = packed_seq_params.max_seqlen_k if hasattr(packed_seq_params, 'max_seqlen_k') else max_seqlen_q
        
        # Get total tokens from cu_seqlens (last element is the total)
        # cu_seqlens format: [0, len_seq1, len_seq1+len_seq2, ..., total_tokens]
        total_q = cu_seqlens_q[-1].item()
        total_k = cu_seqlens_k[-1].item()
        
        # Reshape to unpadded format: (total_tokens, num_heads_per_partition, head_dim)
        # Input is in (seq_len, batch_size, ...) format, flatten first two dims
        q_unpad = q.view(-1, self.num_attention_heads_per_partition, self.head_dim)[:total_q].contiguous()
        k_unpad = k.view(-1, self.num_query_groups_per_partition, self.head_dim)[:total_k].contiguous()
        v_unpad = v.view(-1, self.num_query_groups_per_partition, self.head_dim)[:total_k].contiguous()
        
        # Stage 1: Top-K selection (if enabled)
        topk_idx = None
        if self.use_stage1:
            # Ensure cu_seqlens are on the correct device and dtype
            cu_seqlens_q = cu_seqlens_q.to(device=q_unpad.device, dtype=torch.int32)
            cu_seqlens_k = cu_seqlens_k.to(device=k_unpad.device, dtype=torch.int32)
            
            # =====================================================================
            # Validate inputs before Stage1 to prevent numerical issues
            # =====================================================================
            # Check for Inf/NaN in Q/K/V - these will cause Stage1 to output Inf
            # We only handle NaN/Inf, relying on QK LayerNorm for numerical stability
            # We monitor large values with a threshold but don't clamp them
            
            # Threshold for monitoring large values (no clamping)
            clamp_threshold = 50.0
            
            # Q value validation
            if q_unpad.isnan().any() or q_unpad.isinf().any():
                if torch.distributed.get_rank() == 0:
                    print(f"[InfLLMV2] CRITICAL: q_unpad contains NaN/Inf before Stage1!")
                    print(f"[InfLLMV2] NaN: {q_unpad.isnan().sum().item()}, Inf: {q_unpad.isinf().sum().item()}")
                    print(f"[InfLLMV2] q_unpad stats: min={q_unpad.min().item():.6f}, max={q_unpad.max().item():.6f}, mean={q_unpad.mean().item():.6f}")
                # Only handle NaN/Inf, don't clamp large values
                q_unpad = torch.nan_to_num(q_unpad, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # Log large Q values for monitoring, but don't clamp
                max_val = q_unpad.abs().max().item()
                if max_val > clamp_threshold:
                    if torch.distributed.get_rank() == 0:
                        print(f"[InfLLMV2] WARNING: q_unpad has large values (max_abs={max_val:.2f}), threshold={clamp_threshold:.2f} (no clamp)")
            
            # K value validation
            if k_unpad.isnan().any() or k_unpad.isinf().any():
                if torch.distributed.get_rank() == 0:
                    print(f"[InfLLMV2] CRITICAL: k_unpad contains NaN/Inf before Stage1!")
                    print(f"[InfLLMV2] NaN: {k_unpad.isnan().sum().item()}, Inf: {k_unpad.isinf().sum().item()}")
                # Only handle NaN/Inf, don't clamp large values
                k_unpad = torch.nan_to_num(k_unpad, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # Log large K values for monitoring, but don't clamp
                max_val = k_unpad.abs().max().item()
                if max_val > clamp_threshold:
                    if torch.distributed.get_rank() == 0:
                        print(f"[InfLLMV2] WARNING: k_unpad has large values (max_abs={max_val:.2f}), threshold={clamp_threshold:.2f} (no clamp)")
            
            # V value validation
            if v_unpad.isnan().any() or v_unpad.isinf().any():
                if torch.distributed.get_rank() == 0:
                    print(f"[InfLLMV2] CRITICAL: v_unpad contains NaN/Inf before Stage1!")
                    print(f"[InfLLMV2] NaN: {v_unpad.isnan().sum().item()}, Inf: {v_unpad.isinf().sum().item()}")
                    print(f"[InfLLMV2] v_unpad stats: min={v_unpad.min().item():.6f}, max={v_unpad.max().item():.6f}, mean={v_unpad.mean().item():.6f}")
                # Only handle NaN/Inf, don't clamp large values
                v_unpad = torch.nan_to_num(v_unpad, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # Log large V values for monitoring, but don't clamp
                max_val = v_unpad.abs().max().item()
                if max_val > clamp_threshold:
                    if torch.distributed.get_rank() == 0:
                        print(f"[InfLLMV2] WARNING: v_unpad has large values (max_abs={max_val:.2f}), threshold={clamp_threshold:.2f} (no clamp)")
            
            # =====================================================================
            # Stage 1: Top-K Context Selection (Block Scoring)
            # =====================================================================
            # 
            # According to official implementation (infllmv2_sparse_attention.py):
            # 
            # Stage1 kernel performs the following operations:
            # 1. Reshapes Q from (total_q, nheads, head_dim) to (total_q * group_size, nheads_k, head_dim)
            #    - This expands each query token into group_size virtual tokens
            #    - Example: (7552, 16, 128) → (7552*16, 1, 128) = (120832, 1, 128)
            # 
            # 2. Computes attention scores between expanded Q and K
            #    - Returns raw scores: (nheads_k, total_q * group_size, max_seqlen_k)
            #    - Example: (1, 120832, max_seqlen_k)
            # 
            # 3. External aggregation (this implementation):
            #    - Reshape to (nheads_k, group_size, total_q, max_seqlen_k)
            #    - Sum across group_size dimension → (nheads_k, total_q, max_seqlen_k)
            #    - Example: (1, 16, 7552, max_seqlen_k).sum(dim=1) → (1, 7552, max_seqlen_k)
            # 
            # This aggregated score is used for Top-K block selection.
            
            stage1_scores_raw = infllmv2_attn_stage1(
                q_unpad,  # (total_q, 16, 128) - Pass FULL Q (no pooling!)
                k_unpad,  # (total_k, 1, 128) - Single KV head
                v_unpad,  # (total_k, 1, 128)
                cu_seqlens_q,
                cu_seqlens_k,
                cu_seqlens_k,  # cu_seqlens_v
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                causal=True,
                return_attn_probs=True,  # Required for Top-K selection
            )
            
            # =====================================================================
            # Stage 1 Score Processing
            # =====================================================================
            # 
            # CRITICAL FINDING: Despite what the Python wrapper code suggests,
            # the actual compiled CUDA kernel returns (nheads_k, total_q, max_seqlen_k)
            # directly, WITHOUT expansion. This was confirmed by runtime logs.
            # 
            # The GitHub Python code shows Q expansion logic, but this appears to be
            # internal to the kernel. The kernel returns per-token scores already.
            # 
            # We use the Stage1 output directly without any aggregation.
            
            stage1_scores = stage1_scores_raw.contiguous()
            
            # Follow official implementation: replace NaN with 0 (infllmv2_sparse_attention.py line 589)
            stage1_scores = torch.where(torch.isnan(stage1_scores), torch.zeros_like(stage1_scores), stage1_scores)
            
            if torch.distributed.get_rank() == 0:
                print(f"[InfLLMV2] Stage1 output shape: {stage1_scores.shape} (expecting nheads_k={self.num_query_groups_per_partition}, total_q={total_q})")
            
            # Check for Inf in stage1_scores (NaN already handled above)
            if stage1_scores.isinf().any():
                if torch.distributed.get_rank() == 0:
                    print(f"[InfLLMV2] WARNING: stage1_scores contains Inf!")
                    print(f"[InfLLMV2] Inf count: {stage1_scores.isinf().sum().item()}")
                    print(f"[InfLLMV2] This indicates numerical overflow - inputs may be too large")
                # Replace Inf with 0 (safer than -inf for probabilities)
                # If it's posinf, it was likely overflow, so 0 is reasonable
                # If it's neginf, it was likely masked, so 0 is also reasonable
                stage1_scores = torch.where(torch.isinf(stage1_scores), torch.zeros_like(stage1_scores), stage1_scores)
            
            # =====================================================================
            # Top-K Selection (Performed Outside Kernel)
            # =====================================================================
            # 
            # Official docs state: "Top-K selection should be performed on the 
            # returned aggregated scores (This step is not part of the kernel)"
            #
            # We select top-k blocks from the aggregated scores
            # Now stage1_scores has correct shape: (nheads_k, total_q, max_seqlen_k)
            topk_idx = self._generate_topk_indices(stage1_scores, cu_seqlens_q, max_seqlen_k)
            
            # Ensure valid block indices
            topk_idx = topk_idx.contiguous()
            max_block_idx = (max_seqlen_k + self.block_size - 1) // self.block_size
            topk_idx = torch.clamp(topk_idx, 0, max_block_idx - 1)
            
            if torch.distributed.get_rank() == 0:
                print(f"[InfLLMV2] Top-K indices shape: {topk_idx.shape} (nheads_k={self.num_query_groups_per_partition}, total_q={total_q}, topk_blocks={topk_idx.shape[-1]})")
        
        # =====================================================================
        # Stage 2: Sparse Attention Computation
        # =====================================================================
        # 
        # According to official docs:
        # - "Attention calculation on selected blocks"
        # - "Support for both forward and backward passes"
        # - "Efficient memory access through block-sparse patterns"
        #
        # Input:
        # - q_unpad: (total_q, num_attention_heads, head_dim) - Full queries
        # - k_unpad, v_unpad: Keys and values for all tokens
        # - topk_idx: (nheads_k, total_q, num_topk_blocks) - Selected blocks
        #
        # The kernel computes attention only on the selected blocks,
        # significantly reducing computation for long contexts.
        
        # =====================================================================
        # Pre-Stage2 Validation: Check for extreme values that might cause hang
        # =====================================================================
        # Large V values can cause Stage2 kernel to hang or timeout
        # We log warnings but don't clamp (testing input_layernorm stability)
        v_max_abs = v_unpad.abs().max().item()
        if v_max_abs > 100.0:  # Very large threshold for monitoring
            if torch.distributed.get_rank() == 0:
                print(f"[InfLLMV2] WARNING: v_unpad has VERY large values (max_abs={v_max_abs:.2f}) before Stage2")
                print(f"[InfLLMV2] This may cause Stage2 kernel to hang or timeout")
                print(f"[InfLLMV2] v_unpad stats: min={v_unpad.min().item():.6f}, max={v_unpad.max().item():.6f}, mean={v_unpad.mean().item():.6f}")
        
        # Check Q/K values as well
        q_max_abs = q_unpad.abs().max().item()
        k_max_abs = k_unpad.abs().max().item()
        if q_max_abs > 100.0 or k_max_abs > 100.0:
            if torch.distributed.get_rank() == 0:
                print(f"[InfLLMV2] WARNING: Q/K have large values before Stage2: q_max={q_max_abs:.2f}, k_max={k_max_abs:.2f}")
        
        try:
            out_unpad = infllmv2_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=self.attention_dropout,
                softmax_scale=None,  # Will use default 1/sqrt(head_dim)
                causal=True,
                window_size=(-1, -1),  # No window restriction
                softcap=0.0,  # No softcap
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False,
                block_table=None,
                topk_idx=topk_idx,  # Sparse block selection from Stage1
            )
        except Exception as e:
            if torch.distributed.get_rank() == 0:
                print(f"[InfLLMV2] ERROR in infllmv2_attn_varlen_func: {e}")
                print(f"[InfLLMV2] Input shapes: q={q_unpad.shape}, k={k_unpad.shape}, v={v_unpad.shape}")
            raise
        
        # Ensure output is contiguous
        out_unpad = out_unpad.contiguous()
        
        # Validate attention output for numerical stability
        if out_unpad.isnan().any() or out_unpad.isinf().any():
            if torch.distributed.get_rank() == 0:
                print(f"[InfLLMV2] ERROR: out_unpad contains NaN or Inf after attention!")
                print(f"[InfLLMV2] NaN count: {out_unpad.isnan().sum().item()}")
                print(f"[InfLLMV2] Inf count: {out_unpad.isinf().sum().item()}")
                print(f"[InfLLMV2] This likely indicates a bug in InfLLM V2 integration or kernel")
            # Replace with zeros to prevent propagation (emergency fallback)
            out_unpad = torch.nan_to_num(out_unpad, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape output back to (seq_len, batch_size, hidden_size_per_partition)
        # out_unpad shape: (total_q, num_attention_heads_per_partition, head_dim)
        expected_total = seq_len * batch_size
        
        # Flatten heads dimension: (total_q, hidden_size_per_partition)
        out_flat = out_unpad.view(total_q, self.hidden_size_per_partition)
        
        if total_q < expected_total:
            # Pad with zeros if there's padding in the original input
            pad_size = expected_total - total_q
            out_padded = torch.cat([
                out_flat,
                torch.zeros(pad_size, self.hidden_size_per_partition, 
                           dtype=out_flat.dtype, device=out_flat.device)
            ], dim=0)
            out = out_padded.view(seq_len, batch_size, self.hidden_size_per_partition)
        else:
            # No padding needed, direct reshape
            out = out_flat.view(seq_len, batch_size, self.hidden_size_per_partition)
        
        # Output projection (returns output, bias when skip_bias_add=True)
        output, bias = self.dense(out)
        if bias is not None:
            output = output + bias
        
        # Ensure output is contiguous
        output = output.contiguous()
        
        # Final validation for numerical stability
        if output.isnan().any() or output.isinf().any():
            if torch.distributed.get_rank() == 0:
                print(f"[InfLLMV2] ERROR: Final output contains NaN or Inf!")
                print(f"[InfLLMV2] This indicates a serious bug in the attention implementation")
            # Replace with zeros (emergency fallback, but training will likely fail)
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        
        return output, None


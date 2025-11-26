#!/bin/bash

# Qwen3-Next 2B-A0.5B baseline configuration (WITHOUT InfLLM V2)
# Using same architecture as infllmv2 version for comparison:
# - 16 attention heads, 1 KV head (MQA)
# - hidden_size=2048 (head_dim=128)
#
# Purpose: Verify if the numerical instability is caused by:
# 1. InfLLM V2 integration bug
# 2. Training from scratch with this architecture
# 3. General training hyperparameters

source scripts/models/layer_types.sh

FIRST_K_DENSE_REPLACE=28
NLAYERS=28

declare -a arr=()
# 所有层都使用 full_attention（不使用 InfLLM V2）
for ((i = 0; i < NLAYERS; i++)); do
    if (( i < FIRST_K_DENSE_REPLACE )); then
        arr+=(1)
    else
        arr+=(0)
    fi
done

# 将数组转换为逗号分隔的字符串，并用方括号括起来
MOE_LAYER_FREQ=$(IFS=', '; echo "${arr[*]}")
printf -v MOE_LAYER_FREQ '[%s]' "$MOE_LAYER_FREQ"

MODEL_ARGS=(
   # ⚠️ BASELINE: Standard FlashAttention (no InfLLM V2)
   --spec "slime_plugins.models.qwen3_next" "get_qwen3_next_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   
   # ⚠️ Same architecture as InfLLM V2 version for comparison
   --num-attention-heads 16       # 16 Q heads (same as infllmv2)
   --num-query-groups 1           # 1 KV head - MQA (same as infllmv2)
   --kv-channels 128              
   --num-layers 28
   --hidden-size 2048             # DOUBLED from 1024 (for head_dim=128)
   --ffn-hidden-size 6144         # DOUBLED from 3072
   
   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-06
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 151936

   --rotary-base 10000000

   # moe - DOUBLED sizes
   --moe-ffn-hidden-size 1024            # 512 * 2
   --moe-shared-expert-intermediate-size 1024  # 512 * 2
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 2
   --moe-layer-freq $MOE_LAYER_FREQ
   --num-experts 32
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0.001
   
   # NO InfLLM V2 parameters - using standard FlashAttention
)



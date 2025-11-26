# NLAYERS=18
# FIRST_K_DENSE_REPLACE=0

# arr=()
# for ((i=0; i<NLAYERS; i++)); do
#   if (( i < FIRST_K_DENSE_REPLACE )); then
#     arr+=(0)
#   else
#     arr+=(1)
#   fi
# done

# printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"


# MODEL_ARGS=(
#    # 使用 InfLLM V2 版本的 spec
#    --spec "slime_plugins.models.qwen3_next_infllmv2" "get_qwen3_next_infllmv2_spec"

#    --disable-bias-linear
#    --qk-layernorm
#    --group-query-attention
   

#    # - head_dim: 128 ✓
#    --num-attention-heads 16       # 16 query heads
#    --num-query-groups 1            # 16:1 ratio (MQA: 16 query heads, 1 KV head)
#    --kv-channels 128               # KV channels
#    --num-layers 18                 # Reduced from 28 (more balanced)
#    --hidden-size 2048              # head_dim = 2048/16 = 128
#    --ffn-hidden-size 6144          # Scaled proportionally (2048 * 3 = 6144)

#    # InfLLM V2 sparse attention parameters
#    # Note: InfLLM V2 kernel requirements may need verification with this config
#    --infllmv2-topk-blocks 16    # Number of top-k blocks to select in Stage 1
#    --infllmv2-block-size 64       # Block size for sparse attention (typically 64)
#    --infllmv2-use-stage1 true     # Enable Stage 1 sparse attention
#    --infllmv2-use-for-linear-attention false  # Keep false for linear attention layers

#    --normalization RMSNorm
#    --apply-layernorm-1p
#    --position-embedding-type rope
#    --norm-epsilon 1e-06
#    --rotary-percent 0.25
#    --swiglu
#    --untie-embeddings-and-output-weights
#    --vocab-size 151936

#    --rotary-base 10000000

#    # moe
#    --moe-ffn-hidden-size 320      # Reduced from 512
#    --moe-shared-expert-intermediate-size 384  # Reduced from 512
#    --moe-router-score-function softmax
#    --moe-token-dispatcher-type alltoall
#    --moe-router-topk 2
#    --moe-layer-freq $MOE_LAYER_FREQ
#    --num-experts 48               # Increased to balance total/activated params
#    --moe-grouped-gemm
#    --moe-token-drop-policy probs
#    --moe-router-dtype fp32
#    --moe-permute-fusion
#    --moe-aux-loss-coeff 0.001
#   #  --post_self_attn_layernorm
#   #  --post_mlp_layernorm
# )

# # Parameter summary:
# # - Total params: ~2.52B (~2.5B ✓)
# # - Activated params: ~0.89B (~0.7B, slightly higher but reasonable)
# # - Activation ratio: ~35.4% (reasonable, not too high)
# # - Ratio: 16:1 (num_attention_heads:num_query_groups) ✓
# # - head_dim: 128 ✓

NLAYERS=28
FIRST_K_DENSE_REPLACE=0

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"


MODEL_ARGS=(
   # 使用 InfLLM V2 版本的 spec
   --spec "slime_plugins.models.qwen3_next_infllmv2" "get_qwen3_next_infllmv2_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   
   # ⚠️⚠️⚠️ CRITICAL: Modified to match InfLLM V2 kernel requirements ⚠️⚠️⚠️
   # InfLLM V2 kernel only supports:
   # - MQA (16/1): 16 query heads, 1 KV head
   # - head_dim = 128
   #
   # Original config: 16 Q heads, 2 KV heads (GQA), head_dim=64
   # Modified config: 16 Q heads, 1 KV head (MQA), head_dim=128
   # 
   # ⚠️ Note: This changes the model architecture!
   # ⚠️ Cannot load weights from models trained with different head configs
   --num-attention-heads 16       # Keep 16 (matches kernel requirement: 16/1)
   --num-query-groups 1           # Changed from 2 to 1 (MQA instead of GQA)
   --kv-channels 128              # Kept same
   --num-layers 28
   --hidden-size 2048             # Changed from 1024 to 2048 (for head_dim = 2048/16 = 128)
   --ffn-hidden-size 4096         # Scaled proportionally (3072 * 2 = 6144)

   # InfLLM V2 sparse attention parameters
   # ✅ Now using compatible configuration!
   --infllmv2-topk-blocks 16      # Number of top-k blocks to select in Stage 1
   --infllmv2-block-size 64       # Block size for sparse attention (typically 64)
   --infllmv2-use-stage1 true     # ✅ ENABLED with compatible config (8/1 MQA, head_dim=128)
   --infllmv2-use-for-linear-attention false  # Keep false for linear attention layers

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-06
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 151936

   --rotary-base 10000000

   # moe
   --moe-ffn-hidden-size 512
   --moe-shared-expert-intermediate-size 512
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
  #  --post_self_attn_layernorm
  #  --post_mlp_layernorm
)

# vocab size 151936 * 384 = 58M
# attention w -> 


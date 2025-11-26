NLAYERS=28
FIRST_K_DENSE_REPLACE=0

# 设置 MoE 层频率 - 所有层都使用 MoE
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
   --spec "slime_plugins.models.gpt_oss" "get_gpt_oss_spec"

   # Architecture: GptOssForCausalLM
   # Model parameters: ~2B with 0.5B active
   
   # Attention configuration
   # attention_bias is true, so we don't use --disable-bias-linear
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 2
   --kv-channels 128
   --disable-bias-linear
   
   # Layer configuration
   --num-layers 28
   --hidden-size 1024
   --ffn-hidden-size 1024
   
   # Layer types: alternating sliding_attention and full_attention
   # Layer 0: sliding_attention, Layer 1: full_attention, etc.
   # Total 28 layers with alternating pattern
   
   # Normalization
   --normalization RMSNorm
   --apply-layernorm-1p
   --norm-epsilon 1e-05
   
   # Position embedding and RoPE configuration
   --position-embedding-type rope
   --rotary-percent 1.0
   --rotary-base 150000
   --max-position-embeddings 131072
   
   # RoPE Scaling (YARN style)
   # rope_type: yarn
   # factor: 32.0
   # beta_fast: 32.0
   # beta_slow: 1.0
   # original_max_position_embeddings: 4096
   --use-rope-scaling
   --rope-scaling-factor 32.0
   
   # Sliding window attention
   # sliding_window: 128
   # window_attn_skip_freq: 2 (alternating pattern)
   # softmax_type: learnable
   # Note: These are hardcoded in model_patch.py (lines 49-51):
   #   kw_args['window_size'] = (128, 0)
   #   kw_args['softmax_type'] = "learnable"
   #   kw_args['window_attn_skip_freq'] = 2
   # DO NOT pass as command-line arguments - they're built into GPT-OSS spec
   
   # Activation and embeddings
   # hidden_act: quick_gelu (hardcoded in model_patch.py)
   # NOTE: Do NOT use --swiglu or any other activation flags
   # The activation_func is set by deepmega model_patch.py line 49:
   #   kw_args['activation_func'] = quick_gelu
   #   kw_args['gated_linear_unit'] = True
   #   kw_args['bias_activation_fusion'] = False (but we also set it via CLI to be safe)
   
   # Explicitly disable bias fusion since quick_gelu doesn't support it
   --no-bias-gelu-fusion

   --untie-embeddings-and-output-weights
   
   # Vocabulary
   --vocab-size 201088
   # eos_token_id: 200002
   # pad_token_id: 199999
   
  # MoE configuration
  # NOTE: The moe_ffn_hidden_size should match intermediate_size from HuggingFace config
  # HuggingFace config has intermediate_size=1024
  # Setting to 512 for MoE expert size
   --moe-ffn-hidden-size 1024
   --num-experts 16
   --moe-router-topk 2
   --moe-layer-freq $MOE_LAYER_FREQ
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0.001
   # output_router_logits: false
   
   # Quantization config (mxfp4)
   # modules_to_not_convert: self_attn, mlp.router, embed_tokens, lm_head
   
   # Other settings
   # use_cache: true
   # tie_word_embeddings: false
   # initial_context_length: 4096
   # initializer_range: 0.02
)

# Model size calculation:
# vocab size 201088 * hidden_size 1024 = ~206M parameters (embeddings)
# Total model parameters: ~2B with 0.5B active (due to MoE with experts_per_token=2)
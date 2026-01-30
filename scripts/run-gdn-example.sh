#!/bin/bash

# ============================================================================
# 查看模型形状 (可选，用于调试)
# ============================================================================
# 在运行训练之前，可以先查看模型的参数形状：
#
# PYTHONPATH=/root/Megatron-LM python tools/print_model_shape.py \
#     --num-layers 6 \
#     --hidden-size 512 \
#     --num-attention-heads 8 \
#     --group-query-attention \
#     --num-query-groups 2 \
#     --vocab-size 151936 \
#     --experimental-attention-variant gated_delta_net \
#     --linear-attention-freq 3 \
#     --linear-conv-kernel-dim 4 \
#     --linear-key-head-dim 64 \
#     --linear-value-head-dim 64 \
#     --linear-num-key-heads 4 \
#     --linear-num-value-heads 8 \
#     --random-init \
#     --show-all-params
#
# 可选参数:
#   --show-all-params    显示所有参数而不是按层类型分组
#   --show-dtype         显示参数数据类型
#   --filter-pattern     按名称模式过滤参数 (支持正则表达式)
#   --output-format      输出格式: table, tree, json
# ============================================================================

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# if base folder not set raise error
if [ -z "${BASE_FOLDER}" ]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
  exit 1
fi

if [ -z "${MASTER_ADDR}" ]; then
  export MASTER_ADDR="127.0.0.1"
fi

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ============================================================================
# GDN (Gated Delta Net) Experimental Attention 参数
# ============================================================================
# experimental-attention-variant: 选择 attention 类型
#   - gated_delta_net: Gated Delta Net (线性注意力)
#   - dsa: Dynamic Sparse Attention
#
# linear-attention-freq: 控制线性注意力层和标准 SDPA 层的比例
#   - 整数 N: 每 N 层中有 (N-1) 层是 Linear Attention，1 层是 SDPA
#   - 列表: 如 "([1]*3+[0]*1)*3" 表示 [1,1,1,0,...], 1=LA层, 0=SDPA层
#
# linear-conv-kernel-dim: GDN conv kernel 维度 (默认 4)
# linear-key-head-dim: Q/K head 维度 (默认 128)
# linear-value-head-dim: V/Gate head 维度 (默认 128)
# linear-num-key-heads: Q/K head 数量 (默认 16)
# linear-num-value-heads: V/Gate head 数量 (默认 16)
# ============================================================================

GDN_ARGS=(
   --experimental-attention-variant gated_delta_net
   --linear-attention-freq 3
   --linear-conv-kernel-dim 4
   --linear-key-head-dim 64
   --linear-value-head-dim 64
   --linear-num-key-heads 4
   --linear-num-value-heads 8
)

# 基础模型参数 (以 GPT-small 为例，请根据实际模型调整)
MODEL_ARGS=(
   --num-layers 6
   --hidden-size 512
   --num-attention-heads 8
   --group-query-attention
   --num-query-groups 2
   --swiglu
   --position-embedding-type rope
   --rotary-percent 0.5
   --no-rope-fusion
   --apply-layernorm-1p
   --normalization RMSNorm
   --norm-epsilon 1e-6
   --disable-bias-linear
   --untie-embeddings-and-output-weights
   --vocab-size 151936
   
   # Gated Attention (可选，用于 Qwen3-Next 等模型)
   --attention-output-gate
   
   # 如果使用 MoE
   # --num-experts 32
   # --moe-ffn-hidden-size 64
   # --moe-router-topk 8
   # --moe-grouped-gemm
)

CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/your-model
   --ref-load ${BASE_FOLDER}/your-model_torch_dist
   --load ${BASE_FOLDER}/your-model_slime/
   --save ${BASE_FOLDER}/your-model_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${BASE_FOLDER}/your-data.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 100
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${BASE_FOLDER}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 8192
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 4e-4
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group gdn-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 16 32
   --sglang-max-running-requests 256
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # 使用 flash attention backend
   --attention-backend flash
   # Transformer Engine 是 GDN 的必须项
   --transformer-impl transformer_engine
   
   # 打印模型形状 (调试用，训练时可移除)
   --print-model-shape
)

# launch the master node of ray in container
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${GDN_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

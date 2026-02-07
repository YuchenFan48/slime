#!/bin/bash

# for rerun the task

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export RAY_DEDUP_LOGS=0 
#export CUDA_VISIBLE_DEVICES="7,6,5,4,3,2,1,0"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export HF_DATASETS_OFFLINE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0

MASTER_ADDR=$1
MASTER_PORT=$2
NNODES=$3
CLUSTER_SIZE=$NNODES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0
USE_FUSED_KERNELS=True

LOCAL_ADDR=$__POD_IP__
if [ -z "$LOCAL_ADDR" ]; then
  echo "Error: __POD_IP__ is not set. Please set it to the local address."
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
   --experimental-attention-variant kda
   --linear-attention-freq 4
   --linear-conv-kernel-dim 4
   --linear-key-head-dim 128
   --linear-value-head-dim 128
   --linear-num-key-heads 16
   --linear-num-value-heads 16
)

# 基础模型参数 (以 GPT-small 为例，请根据实际模型调整)
MODEL_ARGS=(
   --disable-bias-linear
   --qk-layernorm
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 2
   --kv-channels 128
   --num-layers 28
   --hidden-size 1024
   --ffn-hidden-size 3072
   --use-gated-attention

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
   --post-self-attn-layernorm
   --post-mlp-layernorm
   # --mtp-num-layers 1
)

CKPT_ARGS=(
   --hf-checkpoint /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-2B-A0.5B
   --ref-load  /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-2B-A0.5B-torch_dist
   --load /apdcephfs/mnt/cephfs/users/yuchenfan/native-mgt-qwen-kimi
   --save /apdcephfs/mnt/cephfs/users/yuchenfan/native-mgt-qwen-kimi
   --save-interval 2048
   --data-source-path slime.ray.rollout_data_source.RolloutDataSourceMultiFileWithBuffer
   --random-init
   --print-model-shape
   # --gated-delta-net-cp-mode sequence_parallel
   # --qkv-format bshd
   # --micro-batch-size 16
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /apdcephfs/mnt/cephfs/data/final_train_data_parquet_jd_new
   --input-key text
   --rollout-shuffle
   --num-rollout 1000000
   --rollout-batch-size 4096
   --global-batch-size 4096
   --eval-batch-size 512

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
   # --log-file-path logs/train_$(date +%Y%m%d_%H%M%S)
   # --save-debug-train-data ./debug_data/{rollout_id}.pt
)

EVAL_ARGS=(
   --eval-interval 512
   --eval-prompt-data c4 /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/c4/validation_sample_truncated.jsonl pes2o /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/pes2o/validation_sample_truncated.jsonl pile /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/pile/validation_sample_truncated.jsonl s2orc /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/s2orc/validation_sample_truncated.jsonl mmlu-redux /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/mmlu-redux-all_ppl.jsonl mmlu-pro /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/mmlu-pro_all_ppl.jsonl supergpqa /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/SuperGPQA-all_ppl.jsonl gsm8k /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/gsm8k_ppl.jsonl bbh /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/bbh_all_ppl.jsonl mbpp /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/mbpp_ppl.jsonl
) 

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 12800

   # --recompute-granularity full
   # --recompute-method uniform
   # --recompute-num-layers 1
#    --balance-data
#    --use-dynamic-batch-size
#    --max-tokens-per-gpu 10240
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 8e-3
   --lr-decay-style WSD
   --lr-wsd-decay-style exponential
   --lr-wsd-decay-iters 20000
   --lr-warmup-iters 2000
   --lr-decay-iters 1000000
   --min-lr 0
   --adam-beta1 0.9
   --adam-beta2 0.95
   --override-opt_param-scheduler
   # --weight-decay 0.1
   # 其他训练参数，如 --global-batch-size, --train-iters 等
   # --train-iters 10000  # 明确指定总步数
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-pretrain
   --wandb-group qwen-kimi-olmo3
   --wandb-key 448ad9b79b563f75fbc01c9a69db00e98ffadae2
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

MTP_TRAINING_ARGS=(
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.1
   --mtp-num-layers 1
)

# Set PYTHONPATH for the training environment
export PYTHONPATH="/apdcephfs/mnt/cephfs/users/yuchenfan/Megatron-LM/:${SCRIPT_DIR}:$PYTHONPATH"

# Ray cluster management - following the working script pattern
if [ $MASTER_ADDR == $LOCAL_ADDR ]; then
    echo "Starting as master node..."
    ray stop --force
    if [ $NNODES -gt 1 ]; then
        # Multi-node setup - start Ray cluster leader
        bash /apdcephfs/mnt/cephfs/users/yuchenfan/slime_search/examples/search-r1/ray_start.sh leader --ray_port=6379 --ray_cluster_size=$CLUSTER_SIZE --ray_init_timeout=300
    fi
    
    # Run training directly (not using ray job submit)
    python3 train_async.py \
        --actor-num-nodes 4 \
        --actor-num-gpus-per-node 8 \
        ${MODEL_ARGS[@]} \
        ${CKPT_ARGS[@]} \
        ${SFT_ARGS[@]} \
        ${OPTIMIZER_ARGS[@]} \
        ${EVAL_ARGS[@]} \
        ${GRPO_ARGS[@]} \
        ${WANDB_ARGS[@]} \
        ${PERF_ARGS[@]} \
        ${SGLANG_ARGS[@]} \
        ${MISC_ARGS[@]} \
        ${CUSTOM_ARGS[@]} \
        ${MTP_TRAINING_ARGS[@]} \
        ${MHC_ARGS[@]}
else
    echo "Starting as worker node..."
    ray stop --force
    # Worker node - connect to master's Ray cluster
    bash /apdcephfs/mnt/cephfs/users/yuchenfan/slime_search/examples/search-r1/ray_start.sh worker --ray_address=$MASTER_ADDR --ray_port=6379
fi

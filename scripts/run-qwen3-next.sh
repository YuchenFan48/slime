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

MODEL_NAME="qwen3-next"
TOTAL_PARAMS="2B"
ACTIVE_PARAMS="0.5B"  # Inferred from A0.5B in file paths
LEARNING_RATE="4e-3"      # From OPTIMIZER_ARGS
GLOBAL_BATCH_SIZE="2048"   # From SFT_ARGS

WANDB_RUN_NAME="${MODEL_NAME}-${TOTAL_PARAMS}-A${ACTIVE_PARAMS}-lr${LEARNING_RATE}-bsz${GLOBAL_BATCH_SIZE}-kimi"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-kimi-2B-A0.5B.sh"

CKPT_ARGS=(
   --hf-checkpoint /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-2B-A0.5B
   #--hf-checkpoint /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-FP8
   --ref-load /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-2B-A0.5B-torch_dist
   # --load /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/
   # --save /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/
   # --save-interval 4096
)


EVAL_ARGS=(
   --eval-interval 512
   --eval-prompt-data c4 /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/c4/validation_sample_truncated.jsonl pes2o /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/pes2o/validation_sample_truncated.jsonl pile /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/pile/validation_sample_truncated.jsonl s2orc /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/s2orc/validation_sample_truncated.jsonl mmlu-redux /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/mmlu-redux-all_ppl.jsonl mmlu-pro /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/mmlu-pro_all_ppl.jsonl supergpqa /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/SuperGPQA-all_ppl.jsonl gsm8k /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/gsm8k_ppl.jsonl bbh /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/bbh_all_ppl.jsonl mbpp /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/mbpp_ppl.jsonl
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /apdcephfs/mnt/cephfs/users/yuchenfan/final_train_data_parquet
   --input-key text
   --rollout-shuffle
   --num-rollout 1000000
   --rollout-batch-size 1024
   --global-batch-size 1024

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
   # --log-file-path logs/train_$(date +%Y%m%d_%H%M%S)
   # --save-debug-train-data ./debug_data/{rollout_id}.pt
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   # --recompute-granularity full
   # --recompute-method uniform
   # --recompute-num-layers 1
   --balance-data
   --use-dynamic-batch-size
   --max-tokens-per-gpu 11264
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 4e-3
   --lr-decay-style WSD
   --lr-wsd-decay-style exponential
   --lr-wsd-decay-iters 100000
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
        --actor-num-nodes 1 \
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
        ${MTP_TRAINING_ARGS[@]}
else
    echo "Starting as worker node..."
    ray stop --force
    # Worker node - connect to master's Ray cluster
    bash /apdcephfs/mnt/cephfs/users/yuchenfan/slime_search/examples/search-r1/ray_start.sh worker --ray_address=$MASTER_ADDR --ray_port=6379
fi

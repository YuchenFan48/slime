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

MODEL_NAME="qwen3-next"
TOTAL_PARAMS="2B"
ACTIVE_PARAMS="0.5B"  # Inferred from A0.5B in file paths
LEARNING_RATE="1e-3"      # From OPTIMIZER_ARGS
GLOBAL_BATCH_SIZE="1024"   # From SFT_ARGS

WANDB_RUN_NAME="${MODEL_NAME}-${TOTAL_PARAMS}-A${ACTIVE_PARAMS}-lr${LEARNING_RATE}-bsz${GLOBAL_BATCH_SIZE}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-next-2B-A0.5B.sh"

CKPT_ARGS=(
   --hf-checkpoint /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-2B-A0.5B
   #--hf-checkpoint /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-FP8
   --ref-load /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-2B-A0.5B-torch_dist
   --load /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-2B-A0.5B-pretrain-lr-1e-3-50B-lr-decay-head-256-wsd-fix-train-std-linear-min-lr-0/
   --save /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-2B-A0.5B-pretrain-lr-1e-3-50B-lr-decay-head-256-wsd-fix-train-std-linear-min-lr-0-/
   --save-interval 4096
)


EVAL_ARGS=(
   --eval-interval 512
   --eval-prompt-data c4 /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/c4/validation_sample_truncated.jsonl pes2o /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/pes2o/validation_sample_truncated.jsonl pile /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/pile/validation_sample_truncated.jsonl s2orc /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/ppl_data/s2orc/validation_sample_truncated.jsonl
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /apdcephfs/mnt/cephfs/users/yuchenfan/pretraining/train_data/dolmino-mix-1124/processed_data
   --input-key text
   --rollout-shuffle
   --num-rollout 10000
   --rollout-batch-size 1024
   --global-batch-size 1024

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # --recompute-granularity full
   # --recompute-method uniform
   # --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-3
   --lr-decay-style WSD
   --lr-wsd-decay-style exponential
   --lr-wsd-decay-iters 2000
   --lr-warmup-iters 2000
   --lr-decay-iters 10000
   --min-lr 0
   --adam-beta1 0.9
   --adam-beta2 0.98
   # --weight-decay 0.1
   # 其他训练参数，如 --global-batch-size, --train-iters 等
   # --train-iters 10000  # 明确指定总步数
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-pretrain
   --wandb-group "${WANDB_RUN_NAME}" # Use the dynamic name for the specific run
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

# MTP_TRAINING_ARGS=(
#    --enable-mtp-training
#    --mtp-loss-scaling-factor 0.1
#    --mtp-num-layers 2
# )

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
  if [[ "$WORKER_IP" == "$MLP_WORKER_0_HOST" ]]; then
    continue
  fi
  echo "Starting Ray worker on ${WORKER_IP}"
  ssh root@"${WORKER_IP}" \
    "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265" &
done
wait


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
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}  \
   ${MTP_TRAINING_ARGS[@]}

#!/bin/bash
export PARTITION=${GROUP:-"default"}
# 修改成你的路径！！！！！！！！！！！"""
export CFSCTL=/mnt/shared-storage-user/p1-shared/yuchenzhang/cfs/bin/cfsctl
export CFG=/mnt/shared-storage-user/p1-shared/yuchenzhang/cfs/cfsd.cfg
export TORCH_CUDA_ARCH_LIST="9.0"

# Multi-node environment (defaults for single-node if not provided)
export RANK=${NODE_RANK:-0}
export NODE_COUNT=${KUBEBRAIN_REPLICA_TOTAL:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PROC_PER_NODE=${PROC_PER_NODE:-8}

# Replace sources.list with new configuration
tee /etc/apt/sources.list > /dev/null << 'EOF'
# ubuntu 22.04
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy main restricted universe multiverse
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-security main restricted universe multiverse
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-updates main restricted universe multiverse
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-proposed main restricted universe multiverse
deb     http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-backports main restricted universe multiverse
deb-src http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy main restricted universe multiverse
deb-src http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-security main restricted universe multiverse
deb-src http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-updates main restricted universe multiverse
deb-src http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-proposed main restricted universe multiverse
deb-src http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu/ jammy-backports main restricted universe multiverse
EOF

# Update package lists (COMMENTED OUT - dependencies already installed)
apt-get update

DEBIAN_FRONTEND=noninteractive apt-get -y install \
gcc g++ automake cmake libtool pkgconf \
libpmemobj-dev libmemkind-dev libtbb-dev rapidjson-dev \
libjson-c-dev libboost-dev gettext libfuse2 libfuse-dev \
git sudo vim curl libcurl4-openssl-dev wget pandoc \
gfortran bzip2 flex libpmix-dev libnl-3-dev libibverbs-dev libssl-dev \
gdb numactl python3 python3-venv python3-pip binutils-dev

# ========== Clean up existing CFS instance before starting ==========
echo "Checking for existing CFS instances..."
if ps aux | grep -v grep | grep -q "cfsd.*$PARTITION"; then
    echo "Found existing CFS instance, stopping it..."
    $CFSCTL -p $PARTITION -s $CFG stop 2>/dev/null || true
    sleep 2
    # Force kill if still running
    pkill -9 cfsd 2>/dev/null || true
    pkill -9 cfsfuse 2>/dev/null || true
    sleep 1
fi

# Clean up lock files and unmount
rm -f /mnt/shared-storage-user/p1-shared/yuchenzhang/cfs/run/default/default/cfsd-*.lock 2>/dev/null || true
rm -f /mnt/shared-storage-user/p1-shared/yuchenzhang/cfs/run/default/default/server-info-file 2>/dev/null || true
umount -l /nvme/fanyuchen/pretrain 2>/dev/null || true

echo "Starting CFS..."
$CFSCTL -p $PARTITION -n $NODE_COUNT -X $MASTER_ADDR -s $CFG start;   
[ $? -ne 0 ] && exit 1

cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime
pip install -U brainpp -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
pip install -e . --no-deps --no-index --disable-pip-version-check --no-build-isolation -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn

# 10.102.223.30 ：只有rule
# 10.102.208.20 ：部署了sft后的xverify
# 10.102.205.32 ：部署了30b-instruct作为reward model
# pip install tensorboard colorama
# for rerun the task
# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3
# pkill -9 ray
# pkill -9 python
set -ex
export PIP_INDEX_URL="http://mirrors.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export PIP_NO_INDEX="false" # 如果要完全禁用公网访问，改为 "true"




export WANDB_MODE="offline"
export WANDB_KEY="4570037654c4911725795e407a3bdd10642495dd"
export WANDB_DIR="/mnt/shared-storage-user/p1-shared/yuchenzhang/wandb"

EXP_NAME="pretrain-qwen-next-fine-web"

# ========== 角色识别与 MASTER_ADDR 设置 ==========
if [ -z "$RANK" ]; then
  echo "RANK not set. Please set RANK=0 for master, RANK=1,2,... for workers"
  exit 1
fi

SHARED_DIR="/mnt/shared-storage-user/p1-shared/yuchenzhang"
READY_FLAG_FILE="$SHARED_DIR/ray_head_ready_30B"

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

HAS_NVLINK=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/gpt-oss-2b-A0.5B.sh"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

CKPT_ARGS=(
   --hf-checkpoint /mnt/shared-storage-user/p1-shared/yuchenzhang/gpt-oss-2b-A0.5B
   --ref-load /mnt/shared-storage-user/p1-shared/yuchenzhang/gpt-oss-2b-A0.5B-torch_dist
   --load /mnt/shared-storage-user/p1-shared/yuchenzhang/gpt-oss-2b-A0.5B-pretrain-lr-std-loss-mask-yuchenzhang-new-5e-3-exp-hidden-1024-16-newdata/
   --save /mnt/shared-storage-user/p1-shared/yuchenzhang/gpt-oss-2b-A0.5B-pretrain-lr-std-loss-mask-yuchenzhang-new-5e-3-exp-hidden-1024-16-newdata/
   --save-interval 4096
   --no-load-optim  # 跳过优化器状态加载，避免sharding类型不兼容
)


EVAL_ARGS=(
   --eval-interval 512
   --eval-prompt-data c4 /mnt/shared-storage-user/p1-shared/yuchenzhang/eval_data/c4.jsonl pes2o /mnt/shared-storage-user/p1-shared/yuchenzhang/eval_data/pes2o.jsonl pile /mnt/shared-storage-user/p1-shared/yuchenzhang/eval_data/pile.jsonl s2orc /mnt/shared-storage-user/p1-shared/yuchenzhang/eval_data/s2orc.jsonl
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /mnt/shared-storage-user/p1-shared/fanyuchen/pretrain/pretrain/dolmo/processed_data/_merged_dclm
   --input-key text
   --rollout-shuffle
   --num-rollout 100000
   --rollout-batch-size 1024
   --global-batch-size 1024

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
   --log-file-path logs/train_$(date +%Y%m%d_%H%M%S)
  #  --save-debug-train-data debug/rollout_id_{rollout_id}/rank_{rank}.pt
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
   --max-tokens-per-gpu 9216
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-3
   --lr-decay-style WSD
   --lr-wsd-decay-style exponential
   --lr-wsd-decay-iters 10000
   --lr-warmup-iters 2000
   --lr-decay-iters 100000
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
   --wandb-group "${EXP_NAME}" # Use the dynamic name for the specific run
   --wandb-key 448ad9b79b563f75fbc01c9a69db00e98ffadae2
   --wandb-mode offline
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
   # --enable-mtp-training  # 如果需要启用MTP训练，取消注释此行
   --mtp-loss-scaling-factor 0.1
   --mtp-num-layers 1
)

# ========= 启动 Ray =========
if [ "$RANK" == "0" ]; then
  if [ -f "$READY_FLAG_FILE" ]; then
    rm -f "$READY_FLAG_FILE"
  fi
  echo "[RANK 0] Starting Ray Head node..."
  ray start --head --port=6379 --node-ip-address=$MASTER_ADDR --num-gpus=8 --disable-usage-stats
  echo "[RANK 0] Ray Head started successfully."
  touch "$READY_FLAG_FILE"
else
  echo "[RANK $RANK] Waiting for Ray Head to be ready..."
  sleep 10

  MAX_WAIT=120
  elapsed=0
  while [ ! -f "$READY_FLAG_FILE" ] && [ $elapsed -lt $MAX_WAIT ]; do
    echo "  ⏳ Still waiting... ($elapsed/$MAX_WAIT)"
    sleep 2
    elapsed=$((elapsed + 2))
  done

  if [ ! -f "$READY_FLAG_FILE" ]; then
    echo "❌ Timed out waiting for Ray Head to be ready."
    exit 1
  fi

  WORKER_IP=$(hostname -I | awk '{print $1}')

  echo "[RANK $RANK] Detected Ray Head at $MASTER_ADDR, starting worker at $WORKER_IP..."
  ray start --address=$MASTER_ADDR:6379 --node-ip-address=$WORKER_IP --num-gpus=8 --disable-usage-stats --block
  echo "[RANK $RANK] Worker started successfully."
fi

wait


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/mnt/shared-storage-user/p1-shared/yuchenzhang/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

if [ "$RANK" == "0" ]; then
    cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime
    ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes ${NODE_COUNT} \
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
fi

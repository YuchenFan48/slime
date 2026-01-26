#!/bin/bash
export PARTITION=${GROUP:-"default"}
# 修改成你的路径！！！！！！！！！！！"""
export CFSCTL=/mnt/shared-storage-user/p1-shared/yuchenzhang/cfs/bin/cfsctl
export CFG=/mnt/shared-storage-user/p1-shared/yuchenzhang/cfs/cfsd.cfg
# 默认启用 CFS/FUSE（设置为1可跳过）
export SKIP_CFS=${SKIP_CFS:-0}
export TORCH_CUDA_ARCH_LIST="9.0"

# Multi-node environment (defaults for single-node if not provided)
export RANK=${NODE_RANK:-0}
export NODE_COUNT=${KUBEBRAIN_REPLICA_TOTAL:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PROC_PER_NODE=${PROC_PER_NODE:-8}

# Configure apt proxy if http_proxy is set
if [ -n "$http_proxy" ]; then
    echo "Configuring apt to use proxy: $http_proxy"
    mkdir -p /etc/apt/apt.conf.d
    echo "Acquire::http::Proxy \"$http_proxy\";" > /etc/apt/apt.conf.d/99proxy
    echo "Acquire::https::Proxy \"${https_proxy:-$http_proxy}\";" >> /etc/apt/apt.conf.d/99proxy
fi

# Remove all other apt source files that might contain external sources
rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true

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
if [ "${SKIP_CFS:-0}" != "1" ]; then
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

    # Load FUSE kernel module if not already loaded
    echo "Checking FUSE support..."
    FUSE_MODULE_LOADED=false
    FUSE_DEVICE_EXISTS=false

    # Check if FUSE module is loaded
    if lsmod | grep -q "^fuse "; then
        echo "FUSE kernel module is already loaded."
        FUSE_MODULE_LOADED=true
    else
        echo "FUSE kernel module not loaded, attempting to load..."
        # Try loading with modprobe
        if modprobe fuse 2>/dev/null; then
            echo "Successfully loaded FUSE kernel module."
            FUSE_MODULE_LOADED=true
        else
            # Try with sudo if available
            if command -v sudo >/dev/null 2>&1; then
                echo "Trying to load FUSE module with sudo..."
                if sudo modprobe fuse 2>/dev/null; then
                    echo "Successfully loaded FUSE kernel module with sudo."
                    FUSE_MODULE_LOADED=true
                fi
            fi
            if [ "$FUSE_MODULE_LOADED" = false ]; then
                echo "Warning: Failed to load FUSE kernel module."
                echo "This may require root privileges or container capabilities."
            fi
        fi
    fi

    # Check if FUSE device exists
    if [ -e /dev/fuse ]; then
        echo "FUSE device /dev/fuse exists."
        FUSE_DEVICE_EXISTS=true
    else
        echo "Warning: FUSE device /dev/fuse does not exist."
    fi

    # Final check and handle FUSE unavailability
    if [ "$FUSE_MODULE_LOADED" = false ] && [ "$FUSE_DEVICE_EXISTS" = false ]; then
        echo "ERROR: FUSE is not available. CFS requires FUSE to mount the filesystem."
        echo ""
        echo "Solutions:"
        echo "1. If running via rjob, add: --custom-resources brainpp.cn/fuse=1"
        echo "2. If running in a container, use --privileged or --cap-add SYS_MODULE"
        echo "3. If data is already available locally, set SKIP_CFS=1 to skip CFS mount"
        echo ""
        echo "FUSE is required for CFS. Exiting..."
        exit 1
    else
        echo "FUSE support confirmed. Starting CFS..."
        $CFSCTL -p $PARTITION -n $NODE_COUNT -X $MASTER_ADDR -s $CFG start
        CFS_START_EXIT_CODE=$?
        
        if [ $CFS_START_EXIT_CODE -ne 0 ]; then
            echo "ERROR: CFS failed to start (exit code: $CFS_START_EXIT_CODE)"
            exit 1
        fi
        
        # Wait a moment and check if fuse daemon is running
        sleep 2
        if ! pgrep -f "cfsfuse" > /dev/null; then
            echo "WARNING: CFS started but fuse daemon may not be running."
            echo "This might cause issues accessing mounted paths."
        else
            echo "CFS and fuse daemon started successfully."
        fi
    fi
else
    echo "SKIP_CFS=1 已设置，本次运行跳过 CFS/FUSE 挂载。"
fi

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
   --load /mnt/shared-storage-user/p1-shared/yuchenzhang/gpt-oss-2b-A0.5B-0114-newdata-v2/
   --save /mnt/shared-storage-user/p1-shared/yuchenzhang/gpt-oss-2b-A0.5B-0114-newdata-v2/
   --save-interval 2048
   --no-load-optim  # 跳过优化器状态加载，避免sharding类型不兼容
)


EVAL_ARGS=(
   --eval-interval 512
   --eval-prompt-data c4 /mnt/shared-storage-user/p1-shared/yuchenzhang/eval_data/c4.jsonl pes2o /mnt/shared-storage-user/p1-shared/yuchenzhang/eval_data/pes2o.jsonl pile /mnt/shared-storage-user/p1-shared/yuchenzhang/eval_data/pile.jsonl s2orc /mnt/shared-storage-user/p1-shared/yuchenzhang/eval_data/s2orc.jsonl
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /nvme/fanyuchen/pretrain/processed_data
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
   --lr 4e-3
   --lr-decay-style WSD
   --lr-wsd-decay-style exponential
   --lr-wsd-decay-iters 10000
   --lr-warmup-iters 2000
   --lr-decay-iters 100000
   --min-lr 0
   --adam-beta1 0.9
   --adam-beta2 0.95
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

# n n

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

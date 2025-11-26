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

# Update package lists
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

# ========== 配置环境变量 ==========
set -ex
export PIP_INDEX_URL="http://mirrors.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export PIP_NO_INDEX="false"

export WANDB_MODE="offline"
export WANDB_KEY="4570037654c4911725795e407a3bdd10642495dd"
export WANDB_DIR="/mnt/shared-storage-user/p1-shared/yuchenzhang/wandb"

# ========== 安装纯 Python 包 (slime) ==========
# 注意: 不要在这里安装 infllmv2_cuda_impl，因为本地环境可能没有 torch
# infllmv2_cuda_impl 将在 Ray head 启动后（有 torch 的容器中）编译
echo "=== Installing slime (pure Python) ==="
cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime
pip install -U brainpp || echo "⚠ Warning: brainpp installation failed"
pip install -e . --no-deps

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
source "${SCRIPT_DIR}/models/qwen3-next-2B-A0.5B.sh"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

CKPT_ARGS=(
   --hf-checkpoint /mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-A0.5B
   --ref-load /mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-A0.5B-torch_dist
   --load /mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-A0.5B-pretrain-lr-std-loss-mask-yuchenzhang/
   --save /mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-A0.5B-pretrain-lr-std-loss-mask-yuchenzhang/
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
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

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
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-pretrain
   --wandb-group "${EXP_NAME}"
   --wandb-key 448ad9b79b563f75fbc01c9a69db00e98ffadae2
   --wandb-mode offline
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

MTP_TRAINING_ARGS=(
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
  
  # ========== 在 Ray 容器中设置 infllmv2_cuda_impl ==========
  # 这里有 torch，可以成功编译 CUDA 扩展
  echo "=========================================="
  echo "Setting up infllmv2_cuda_impl in Ray container"
  echo "=========================================="
  cd /mnt/shared-storage-user/p1-shared/yuchenzhang/infllmv2_cuda_impl
  
  # 1. 初始化 git submodules (cutlass)
  if [ ! -f "csrc/cutlass/include/cutlass/numeric_types.h" ]; then
      echo "→ Initializing cutlass submodule..."
      git submodule update --init --recursive || {
          echo "⚠ Warning: Failed to initialize git submodules"
          echo "   Trying to clone cutlass manually..."
          rm -rf csrc/cutlass
          git clone --depth 1 https://github.com/NVIDIA/cutlass.git csrc/cutlass || {
              echo "❌ ERROR: Failed to get cutlass library"
              echo "   InfLLM V2 will not work without cutlass"
          }
      }
  else
      echo "✓ Cutlass submodule already initialized"
  fi
  
  # 2. 编译 CUDA 扩展
  if [ -f "infllm_v2/_C.so" ] || ls build/lib*/infllm_v2/_C*.so 2>/dev/null; then
      echo "✓ CUDA extensions already compiled, skipping..."
  else
      echo "→ Compiling CUDA extensions (this may take 5-10 minutes)..."
      python3 setup.py build_ext --inplace || {
          echo "⚠ WARNING: CUDA extension compilation failed"
          echo "   InfLLM V2 attention will not work"
          echo "   You can still run training with standard attention"
      }
      
      if [ -f "infllm_v2/_C.so" ]; then
          echo "✓ CUDA extensions compiled successfully!"
      fi
  fi
  
  cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime
  echo "=========================================="
  echo "Setup completed"
  echo "=========================================="
  
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

# ========== 构建 Ray Runtime Environment ==========
# 添加 infllmv2_cuda_impl 和 slime 到 PYTHONPATH，直接 import 无需 pip install
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/mnt/shared-storage-user/p1-shared/yuchenzhang/Megatron-LM/:/mnt/shared-storage-user/p1-shared/yuchenzhang/infllmv2_cuda_impl:/mnt/shared-storage-user/p1-shared/yuchenzhang/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

# ========== 提交训练任务 ==========
if [ "$RANK" == "0" ]; then
    cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime
    echo "=========================================="
    echo "Submitting training job to Ray"
    echo "=========================================="
    ray job submit --address="http://127.0.0.1:8265" \
    --working-dir=/mnt/shared-storage-user/p1-shared/yuchenzhang/slime \
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

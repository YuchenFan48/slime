#!/bin/bash

set -ex

SCRIPT=/mnt/shared-storage-user/p1-shared/fanyuchen/pretrain/pretrain/slime/scripts/run-qwen3-next-h.sh
NODE_COUNT=4
export KUBEBRAIN_CLUSTER_ENTRY="https://h.pjlab.org.cn"
export KUBEBRAIN_NAMESPACE="ailab-p1"
export BRAINPP_ACCESS_KEY_ID="876f7f3814c85b36e93c132f9afa3b08"
export BRAINPP_SECRET_ACCESS_KEY="806bea178e72c9aefa653ed72a4cd060"
# pip install -U brainpp
# 注意：
# 这里的brainpp aksk请到上海智算平台“密钥管理”处创建权限使用范围为“使用 RJob”的aksk
# --task-type idle \  --charged-group=p1_gpu \
# --negative-tags node/gpu-lg-cmc-h-h200-0979.host.h.pjlab.org.cn \


rjob submit --name=$RUN_NAME-$(date "+%Y%m%d-%H%M%S") \
 --gpu=8 --memory=1800000 --cpu=128 \
 --charged-group=p1_gpu \
 --private-machine=group \
 --mount=gpfs://gpfs1/p1-shared:/mnt/shared-storage-user/p1-shared \
 --image=registry.h.pjlab.org.cn/ailab-p1-p1_gpu/slime:20251022-v3 \
 -P $NODE_COUNT --host-network=true \
 --store-host-nvme \
 --custom-resources mellanox.com/mlnx_rdma=1 \
 --custom-resources rdma/mlnx_shared=8 \
 --custom-resources brainpp.cn/fuse=1 \
 -e DISTRIBUTED_JOB=true \
 -e GROUP=ailab-p1 \
 -- bash $SCRIPT
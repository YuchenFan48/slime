#!/bin/bash

set -ex

SCRIPT=/mnt/shared-storage-user/p1-shared/yuchenzhang/slime/scripts/run-qwen3-next-inf-reduced.sh
NODE_COUNT=1
export KUBEBRAIN_CLUSTER_ENTRY="https://h.pjlab.org.cn"
export KUBEBRAIN_NAMESPACE="ailab-p1"
export BRAINPP_ACCESS_KEY_ID="46d5d1380639cb4bcf736a00df6cb9c8"
export BRAINPP_SECRET_ACCESS_KEY="28c9c43039a292fdf105caf42c5eaa60"

# Skip proxy for internal domains (both uppercase and lowercase)
export NO_PROXY="10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,127.0.0.1,localhost,.pjlab.org.cn,.h.pjlab.org.cn"
export no_proxy="10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,127.0.0.1,localhost,.pjlab.org.cn,.h.pjlab.org.cn"
# pip install -U brainpp
# 注意：
# 这里的brainpp aksk请到上海智算平台"密钥管理"处创建权限使用范围为"使用 RJob"的aksk
# --task-type idle \  --charged-group=p1_gpu \
# --negative-tags node/gpu-lg-cmc-h-h200-0979.host.h.pjlab.org.cn \

# Temporarily disable proxy for rjob submission to avoid 407 error
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

rjob submit --name=$RUN_NAME-$(date "+%Y%m%d-%H%M%S")-inf-reduced-16sparse \
 --gpu=8 --memory=1200000 --cpu=100 \
 --charged-group=p1_gpu \
 --private-machine=group \
 --mount=gpfs://gpfs1/p1-shared:/mnt/shared-storage-user/p1-shared \
 --image=registry.h.pjlab.org.cn/ailab-p1-p1_gpu/infllm:20251112224720 \
 -P $NODE_COUNT --host-network=true \
 --store-host-nvme \
 --custom-resources mellanox.com/mlnx_rdma=1 \
 --custom-resources rdma/mlnx_shared=8 \
 --custom-resources brainpp.cn/fuse=1 \
 -e DISTRIBUTED_JOB=true \
 -e GROUP=ailab-p1 \
 -- bash $SCRIPT
#!/bin/bash
# 前台交互模式 - 会直接进入容器
rlaunch --gpu=8 --memory=18000 --cpu=100 \
 --charged-group=p1_gpu \
 --private-machine=yes \
 --mount=gpfs://gpfs1/p1-shared:/mnt/shared-storage-user/p1-shared \
 --image=registry.h.pjlab.org.cn/ailab-p1-p1_gpu/infllm:20251112224720 \
#  --positive-tags node/gpu-lg-cmc-h-h200-0979.host.h.pjlab.org.cn \
 -- bash 
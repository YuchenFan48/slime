#!/bin/bash

# 转换 torch.distributed checkpoint 到 HuggingFace 格式
# 
# 你的 checkpoint 是 torch.distributed 格式，需要使用 convert_torch_dist_to_hf.py
# 需要通过 rlaunch 在正确的容器环境中运行（包含 megatron.training 等依赖）

# 已经在 rlaunch 容器中，直接运行转换命令
cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime

# 设置必要的环境变量（特别是 PYTHONPATH，确保能找到 Megatron-LM）
export PYTHONPATH="/mnt/shared-storage-user/p1-shared/yuchenzhang/Megatron-LM/:/mnt/shared-storage-user/p1-shared/yuchenzhang/slime:/mnt/shared-storage-user/p1-shared/yuchenzhang/infllmv2_cuda_impl:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python tools/convert_torch_dist_to_hf.py \
    --input-dir /mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-infllmv2-16heads-mqa-hdim128-1120-reduced/iter_0002047 \
    --output-dir /mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-infllmv2-16heads-mqa-hdim128-1120-reduced-hf \
    --model-name qwen3nextinfllmv2 \
    --origin-hf-dir /mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-A0.5B-infllm-reduced \
    --vocab-size 151936 \
    --force

# echo "转换完成！输出目录: /mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-infllmv2-16heads-mqa-hdim128-1120-reduced-hf"

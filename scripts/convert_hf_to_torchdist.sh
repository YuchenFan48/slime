source /mnt/shared-storage-user/p1-shared/yuchenzhang/slime/scripts/models/qwen3-next-2B-A0.5B-infllmv2-reduced.sh

# 注意：MODEL_ARGS 包含所有模型架构参数，包括 --infllmv2-* 参数
# 这些参数是构建模型架构所必需的，不应该被过滤
# 转换脚本会使用这些参数来正确构建 InfLLM V2 模型架构

# 设置必要的环境变量（特别是 PYTHONPATH，确保能找到所有模块）
export PYTHONPATH="/mnt/shared-storage-user/p1-shared/yuchenzhang/Megatron-LM/:/mnt/shared-storage-user/p1-shared/yuchenzhang/slime:/mnt/shared-storage-user/p1-shared/yuchenzhang/infllmv2_cuda_impl:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime

python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-infllmv2-16heads-mqa-hdim128-1120-reduced-hf \
    --model-type qwen3nextinfllm \
    --save /mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-infllmv2-16heads-mqa-hdim128-1120-reduced-hf-torch-dist

source /apdcephfs/mnt/cephfs/users/yuchenfan/slime/scripts/models/qwen3-kimi-2B-A0.5B.sh

PYTHONPATH=/apdcephfs/mnt/cephfs/users/yuchenfan/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /apdcephfs/mnt/cephfs/users/yuchenfan/kimi-test/iter_0000000-hf \
    --save /apdcephfs/mnt/cephfs/users/yuchenfan/kimi-test/iter_0000000-hf-torch-dist 

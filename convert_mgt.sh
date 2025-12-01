PYTHONPATH=/apdcephfs/mnt/cephfs/users/yuchenfan/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-olmo3/iter_0012287 \
  --output-dir /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-kda-olmo3/iter_0012287-hf \
  --origin-hf-dir /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-2B-A0.5B \
  --force

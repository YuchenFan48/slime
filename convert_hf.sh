#!/bin/bash
# Example script for converting HF checkpoint to torch distributed format
# Customize the paths according to your environment

source scripts/models/qwen3-kimi-2B-A0.5B.sh

PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/input-hf-checkpoint \
    --save /path/to/output-torch-dist

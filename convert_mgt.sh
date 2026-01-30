#!/bin/bash
# Example script for converting Megatron checkpoint to HF format
# Customize the paths according to your environment

PYTHONPATH=/path/to/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/megatron-checkpoint \
  --output-dir /path/to/output-hf-checkpoint \
  --origin-hf-dir /path/to/original-hf-model \
  --force

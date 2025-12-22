# HuggingFace → Torch Dist 转换步骤指南

## 快速开始

### 步骤 1: 确认你的模型类型

检查你的 HuggingFace 模型的 `config.json`：

```bash
cat /path/to/your/hf-model/config.json | grep model_type
```

**可能的 model_type：**
- `"qwen3_next"` → 标准 Qwen3 Next（使用 `qwen3_next` bridge）
- `"qwen3nextinfllm"` → Qwen3 Next + InfLLM V2（使用 `qwen3nextinfllm` bridge）
- `"qwen3_kimi"` → Qwen3 Kimi（使用 `qwen3_kimi` bridge）

### 步骤 2: 选择合适的模型配置脚本

根据你的模型类型，选择合适的配置脚本：

```bash
# InfLLM V2 版本
source slime/scripts/models/qwen3-next-2B-A0.5B-infllmv2-reduced.sh

# 标准版本
source slime/scripts/models/qwen3-next-2B-A0.5B.sh

# Kimi 版本
source slime/scripts/models/qwen3-kimi-2B-A0.5B.sh
```

### 步骤 3: 运行转换脚本

#### 方法 A: 使用现有的转换脚本（推荐）

```bash
cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime

# 修改 convert_hf_to_torchdist.sh 中的路径
vim convert_hf_to_torchdist.sh

# 运行
bash convert_hf_to_torchdist.sh
```

#### 方法 B: 手动运行

```bash
cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime

# 1. 加载模型配置
source scripts/models/qwen3-next-2B-A0.5B-infllmv2-reduced.sh

# 2. 运行转换
PYTHONPATH=/mnt/shared-storage-user/p1-shared/yuchenzhang/Megatron-LM \
python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /path/to/your/hf-model \
    --model-type qwen3nextinfllm \
    --save /path/to/output
```

### 步骤 4: 检查输出

转换完成后，检查输出目录：

```bash
ls -lh /path/to/output/
# 应该看到类似这样的文件：
# - iter_0000001/
#   - mp_rank_00_model_states.pt
#   - mp_rank_01_model_states.pt
#   - ...
# - latest_checkpointed_iteration.txt
```

## 完整示例

### 示例 1: 转换 InfLLM V2 模型

```bash
# 设置路径
HF_MODEL="/mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-infllmv2-16heads-mqa-hdim128-1120-reduced-hf"
OUTPUT_DIR="/mnt/shared-storage-user/p1-shared/yuchenzhang/qwen-3-next-2B-infllmv2-16heads-mqa-hdim128-1120-reduced-hf-torch-dist"

# 进入 slime 目录
cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime

# 加载配置
source scripts/models/qwen3-next-2B-A0.5B-infllmv2-reduced.sh

# 运行转换
PYTHONPATH=/mnt/shared-storage-user/p1-shared/yuchenzhang/Megatron-LM \
python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_MODEL}" \
    --model-type qwen3nextinfllm \
    --save "${OUTPUT_DIR}"
```

### 示例 2: 转换标准 Qwen3 Next 模型

```bash
HF_MODEL="/path/to/qwen3-next-hf-model"
OUTPUT_DIR="/path/to/output"

cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime
source scripts/models/qwen3-next-2B-A0.5B.sh

PYTHONPATH=/mnt/shared-storage-user/p1-shared/yuchenzhang/Megatron-LM \
python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_MODEL}" \
    --model-type qwen3_next \
    --save "${OUTPUT_DIR}"
```

## 重要参数说明

### 必需参数

1. **`--hf-checkpoint`**: HuggingFace 模型路径
   ```bash
   --hf-checkpoint /path/to/hf-model
   ```

2. **`--model-type`**: Bridge 类型（必须匹配你的模型）
   ```bash
   --model-type qwen3nextinfllm  # 对于 InfLLM V2 模型
   --model-type qwen3_next       # 对于标准模型
   ```

3. **`--save`**: 输出路径
   ```bash
   --save /path/to/output
   ```

### MODEL_ARGS 包含的参数

`MODEL_ARGS` 会自动包含所有必需的模型架构参数，包括：
- `--spec`: 模型架构定义
- `--num-layers`: 层数
- `--hidden-size`: 隐藏层大小
- `--num-attention-heads`: 注意力头数
- `--infllmv2-*`: InfLLM V2 配置（如果适用）

**你不需要手动指定这些参数，`MODEL_ARGS[@]` 会自动包含它们。**

## 常见问题排查

### 问题 1: `unrecognized arguments` 错误

**原因**: 参数解析问题

**解决**: 确保使用 `${MODEL_ARGS[@]}` 而不是手动列出参数

### 问题 2: `model_type` 不匹配

**原因**: `--model-type` 参数与模型实际类型不匹配

**解决**: 
1. 检查 `config.json` 中的 `model_type`
2. 使用正确的 `--model-type` 参数
3. 或者修改 `config.json` 中的 `model_type`

### 问题 3: 权重名称不匹配

**原因**: Bridge 选择错误或模型架构构建错误

**解决**:
1. 确保 `--model-type` 正确
2. 确保 `MODEL_ARGS` 包含正确的 `--spec` 参数
3. 检查模型配置脚本是否匹配你的模型

### 问题 4: 内存不足

**解决**: 使用多 GPU 并行转换

```bash
# 使用 4 个 GPU
torchrun --nproc-per-node=4 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint ... \
    --model-type ... \
    --save ...
```

## 验证转换结果

转换完成后，可以验证：

```bash
# 检查 checkpoint 文件
ls -lh /path/to/output/iter_0000001/

# 检查 checkpoint tracker
cat /path/to/output/latest_checkpointed_iteration.txt

# 尝试加载（可选）
python -c "
import torch
ckpt = torch.load('/path/to/output/iter_0000001/mp_rank_00_model_states.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('Model keys count:', len(ckpt.get('model', {})))
"
```

## 总结

**最简单的转换方法：**

```bash
cd /mnt/shared-storage-user/p1-shared/yuchenzhang/slime
source scripts/models/qwen3-next-2B-A0.5B-infllmv2-reduced.sh
PYTHONPATH=/mnt/shared-storage-user/p1-shared/yuchenzhang/Megatron-LM \
python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint YOUR_HF_MODEL_PATH \
    --model-type qwen3nextinfllm \
    --save YOUR_OUTPUT_PATH
```

**关键点：**
1. ✅ 使用 `${MODEL_ARGS[@]}` 传递所有参数
2. ✅ `--model-type` 必须匹配你的模型类型
3. ✅ `--hf-checkpoint` 和 `--save` 路径要正确
4. ✅ 确保有足够的磁盘空间和内存


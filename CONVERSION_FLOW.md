# HuggingFace → Megatron 转换流程详解

## 完整转换流程

```
转换脚本执行流程：
┌─────────────────────────────────────────────────────────────┐
│ 1. 解析命令行参数                                            │
│    parse_args(add_convertion_args)                           │
│    ↓                                                          │
│    解析所有 MODEL_ARGS（包括 --infllmv2-* 参数）            │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 设置默认参数                                              │
│    set_default_megatron_args(args)                          │
│    ↓                                                          │
│    设置一些 Megatron 默认值                                  │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 修改 config.json（如果需要）                             │
│    if args.model_type:                                       │
│        config.json["model_type"] = args.model_type          │
│    ↓                                                          │
│    临时修改以选择正确的 bridge                               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 构建 Megatron 模型架构                                    │
│    get_model(get_model_provider_func(args), ...)            │
│    ↓                                                          │
│    get_model_provider_func(args)                             │
│    → 根据 args.spec 找到对应的函数                           │
│    → 例如：get_qwen3_next_infllmv2_spec(args, config)       │
│    ↓                                                          │
│    get_qwen3_next_infllmv2_spec 会：                         │
│    - 读取 args.infllmv2_topk_blocks                         │
│    - 读取 args.infllmv2_block_size                          │
│    - 读取 args.infllmv2_use_stage1                          │
│    - 读取 args.infllmv2_use_for_linear_attention            │
│    - 根据这些参数决定哪些层使用 InfLLM V2                    │
│    - 构建对应的模型架构                                      │
│    ↓                                                          │
│    返回一个空的 Megatron 模型（只有架构，没有权重）         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 创建 Bridge                                               │
│    bridge = AutoBridge.from_pretrained(hf_model_path)      │
│    ↓                                                          │
│    根据 config.json 的 model_type 选择 bridge：              │
│    - "qwen3_next" → Qwen3NextBridge                          │
│    - "qwen3nextinfllm" → Qwen3NextInfLLMV2Bridge            │
│    - "qwen3_kimi" → Qwen3KimiBridge                          │
│    ↓                                                          │
│    Bridge 负责：                                              │
│    - 权重名称映射（HF 名称 ↔ Megatron 名称）                 │
│    - 权重格式转换（HF 格式 ↔ Megatron 格式）                 │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 加载并转换权重                                             │
│    bridge.load_weights(model, hf_model_path)                │
│    ↓                                                          │
│    Bridge 会：                                                │
│    1. 从 HF checkpoint 加载权重                              │
│    2. 根据权重名称映射规则转换名称                           │
│       例如：                                                  │
│       HF: model.layers.0.self_attn.q_proj.weight             │
│       → Megatron: decoder.layers.0.self_attention.query_key_value.weight │
│    3. 根据权重格式转换规则转换格式                           │
│       例如：将 q_proj, k_proj, v_proj 合并为 query_key_value │
│    4. 将转换后的权重加载到 Megatron 模型中                  │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. 保存 Megatron checkpoint                                  │
│    save_checkpoint(1, model, ...)                            │
│    ↓                                                          │
│    保存为 Megatron 格式的 checkpoint                         │
└─────────────────────────────────────────────────────────────┘
```

## 参数分类

### 1. 模型架构参数（必需）

这些参数用于构建模型架构：

```bash
# 基础架构参数
--spec "slime_plugins.models.qwen3_next_infllmv2" "get_qwen3_next_infllmv2_spec"
--num-layers 28
--hidden-size 2048
--num-attention-heads 16
--num-query-groups 1
--ffn-hidden-size 4096
--vocab-size 151936

# InfLLM V2 配置参数（必需！）
--infllmv2-topk-blocks 16          # 用于构建 InfLLM V2 attention
--infllmv2-block-size 64           # 用于构建 InfLLM V2 attention
--infllmv2-use-stage1 true         # 用于构建 InfLLM V2 attention
--infllmv2-use-for-linear-attention false  # 决定哪些层用 InfLLM V2

# MoE 参数
--num-experts 32
--moe-router-topk 2
--moe-ffn-hidden-size 512
```

**为什么需要这些参数？**
- `get_qwen3_next_infllmv2_spec` 函数会读取这些参数
- 根据这些参数决定哪些层使用 InfLLM V2
- 根据这些参数配置 InfLLM V2 的 topk_blocks、block_size 等

### 2. Bridge 选择参数（必需）

```bash
--model-type qwen3nextinfllm  # 告诉 AutoBridge 使用哪个 bridge
```

**为什么需要？**
- AutoBridge 根据 `config.json` 的 `model_type` 选择 bridge
- 不同的 bridge 有不同的权重映射规则
- InfLLM V2 使用 `query_key_value` 格式，标准版本使用 `linear_qgkv` 格式

### 3. 转换专用参数（必需）

```bash
--hf-checkpoint /path/to/hf/model  # HuggingFace 模型路径
--save /path/to/output             # 输出路径
```

### 4. 训练相关参数（不需要，但可以传递）

这些参数在转换时不会被使用，但传递也不会报错（因为有默认值）：

```bash
--train-iters 1000          # 训练迭代次数（转换时不需要）
--lr 1e-4                   # 学习率（转换时不需要）
--batch-size 32             # 批次大小（转换时不需要）
```

## 参数处理策略

### 方案1：传递所有参数（推荐）

```bash
# 直接传递所有 MODEL_ARGS
python convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint ... \
    --model-type qwen3nextinfllm \
    --save ...
```

**优点：**
- 简单，不需要过滤
- 确保所有必需的参数都被传递

**缺点：**
- 会传递一些不需要的参数（但不会报错，因为有默认值）

### 方案2：只传递必需的参数

```bash
# 手动选择需要的参数
python convert_hf_to_torch_dist.py \
    --spec "slime_plugins.models.qwen3_next_infllmv2" "get_qwen3_next_infllmv2_spec" \
    --num-layers 28 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --num-query-groups 1 \
    --infllmv2-topk-blocks 16 \
    --infllmv2-block-size 64 \
    --infllmv2-use-stage1 true \
    --infllmv2-use-for-linear-attention false \
    --hf-checkpoint ... \
    --model-type qwen3nextinfllm \
    --save ...
```

**优点：**
- 只传递必需的参数
- 更清晰

**缺点：**
- 容易遗漏参数
- 维护成本高

## 常见问题

### Q1: 为什么需要 --infllmv2-* 参数？

**A:** 因为 `get_qwen3_next_infllmv2_spec` 函数会读取这些参数来决定：
- 哪些层使用 InfLLM V2 attention
- InfLLM V2 的配置（topk_blocks, block_size 等）

如果这些参数不正确，模型架构就会构建错误，导致权重转换失败。

### Q2: Bridge 需要这些参数吗？

**A:** Bridge 本身不需要这些参数。Bridge 只负责：
- 权重名称映射
- 权重格式转换

但是，如果模型架构构建错误（比如应该用 InfLLM V2 的层用了标准 attention），Bridge 转换权重时就会因为权重名称不匹配而失败。

### Q3: 如果传递了不需要的参数会怎样？

**A:** 不会报错。`parse_args` 会解析所有参数，不需要的参数会被忽略（使用默认值）。

### Q4: 如何知道哪些参数是必需的？

**A:** 查看 `get_qwen3_next_infllmv2_spec` 函数，看它读取了哪些 `args.xxx`。

## 实际转换示例

```bash
# 1. 加载模型配置
source scripts/models/qwen3-next-2B-A0.5B-infllmv2-reduced.sh

# 2. 运行转换（传递所有 MODEL_ARGS）
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /path/to/hf-model \
    --model-type qwen3nextinfllm \
    --save /path/to/output
```

转换过程：
1. 解析所有参数（包括 --infllmv2-*）
2. 构建模型架构（使用 --infllmv2-* 参数）
3. 选择 bridge（根据 --model-type）
4. 转换权重（bridge 负责）
5. 保存 checkpoint


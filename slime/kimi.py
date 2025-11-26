# 文件路径: slime/custom_models.py

from transformers import AutoConfig, AutoTokenizer, Qwen2Config, Qwen2Tokenizer, Qwen2TokenizerFast

class Qwen3KimiConfig(Qwen2Config):
    model_type = "qwen3_kimi"

    def __init__(
        self,
        linear_conv_kernel_dim=4,
        linear_num_value_heads=None,
        linear_num_key_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        attention_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.attention_bias = attention_bias

# 执行注册
try:
    AutoConfig.register("qwen3_kimi", Qwen3KimiConfig)
    AutoTokenizer.register(Qwen3KimiConfig, slow_tokenizer_class=Qwen2Tokenizer, fast_tokenizer_class=Qwen2TokenizerFast)
    print("[Slime/Internal] Successfully registered 'qwen3_kimi' config inside module.")
except Exception as e:
    print(f"[Slime/Internal] Registration failed: {e}")

import sys
sys.path.append("../../")
from config import PATHS

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# 获取本地模型路径
model_path = PATHS["Qwen2.5-7B-Instruct"]
print(f"Loading from local path: {model_path}")

# 先用 transformers 从本地加载 HF 模型
# 先加载到 CPU，然后手动移到 GPU
hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
# 手动将模型移到 GPU
hf_model = hf_model.to("cuda:0")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

print("Converting to HookedTransformer...")

# 用已加载的 hf_model 创建 HookedTransformer
hooked_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    hf_model=hf_model,
    tokenizer=tokenizer,
    dtype=torch.float16,
    device="cuda:0",  # 显式指定设备
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    fold_value_biases=False,  # 禁用 fold_value_biases 避免设备不匹配
)

print("\n===== Model Config =====")
print(hooked_model.cfg)

print("\n===== Key Fields =====")
print("n_layers:", hooked_model.cfg.n_layers)
print("d_model:", hooked_model.cfg.d_model)
print("n_heads:", hooked_model.cfg.n_heads)
print("d_mlp:", hooked_model.cfg.d_mlp)

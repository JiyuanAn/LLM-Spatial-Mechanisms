import sys
sys.path.append("./")
sys.path.append("../../")
from config import PATHS

import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# TRAIN_DATA_FILE_PATH = "../dataGenerate/spatial_reasoning_dataset_ZH.json"
# TEST_DATA_FILE_PATH = "../dataGenerate/spatial_reasoning_dataset_ZH_test.json"
# OUTPUT_FILE_PATH = f"probe_results_{time.strftime('%Y%m%d_%H%M%S')}.json"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str)
parser.add_argument("--train_data_file_path", "-tr", type=str)
parser.add_argument("--test_data_file_path", "-te", type=str)
parser.add_argument("--output_file_path", "-o", type=str, default=f"probe_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
args = parser.parse_args()

MODEL_NAME = args.model_name
TRAIN_DATA_FILE_PATH = args.train_data_file_path
TEST_DATA_FILE_PATH = args.test_data_file_path
OUTPUT_FILE_PATH = args.output_file_path

# =========================
# 1. 基本配置
# =========================
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = "cuda:0"
DTYPE = torch.float16
BATCH_SIZE = 4
RIDGE_ALPHA = 1.0
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# 2. 加载模型
# =========================
print("Loading model...")
print(f"Model path: {MODEL_PATH}")

# 先用 transformers 加载 HF 模型
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True
)
# 手动将模型移到 GPU
hf_model = hf_model.to(DEVICE)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("Converting to HookedTransformer...")

# 转换为 HookedTransformer
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    hf_model=hf_model,
    tokenizer=tokenizer,
    dtype=DTYPE,
    device=DEVICE,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    fold_value_biases=False,
)
model.eval()

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model

print(f"Loaded model with {n_layers} layers, d_model={d_model}")

# =========================
# 3. 示例数据（请替换为你自己的）
# =========================
# 假设你已经有 train / test 数据
# target 是三维向量：[x, y, z]
# x: left (-1) / right (+1)
# y: below (-1) / above (+1)
# z: behind (-1) / front (+1)
# train_data = [
#     {
#         "prompt": "A is left of B. B is above C. Where is A relative to C?",
#         "target": np.array([-1.0, 1.0, 0.0])
#     },
#     {
#         "prompt": "A is right of B. B is below C. Where is A relative to C?",
#         "target": np.array([1.0, -1.0, 0.0])
#     },
#     {
#         "prompt": "A is above B. B is left of C. Where is A relative to C?",
#         "target": np.array([-1.0, 1.0, 0.0])
#     },
#     {
#         "prompt": "A is below B. B is right of C. Where is A relative to C?",
#         "target": np.array([1.0, -1.0, 0.0])
#     },
#     {
#         "prompt": "A is front of B. B is above C. Where is A relative to C?",
#         "target": np.array([0.0, 1.0, 1.0])
#     },
#     {
#         "prompt": "A is behind B. B is below C. Where is A relative to C?",
#         "target": np.array([0.0, -1.0, -1.0])
#     },
#     {
#         "prompt": "A is left of B. B is front of C. Where is A relative to C?",
#         "target": np.array([-1.0, 0.0, 1.0])
#     },
#     {
#         "prompt": "A is right of B. B is behind C. Where is A relative to C?",
#         "target": np.array([1.0, 0.0, -1.0])
#     },
# ]

# test_data = train_data  # demo 用，真实实验请分开

import json
train_data = []
with open(TRAIN_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        train_data.append({
            "prompt": sample["question"],
            "target": np.array(sample["target"])
        })

test_data = []
with open(TEST_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        test_data.append({
            "prompt": sample["question"],
            "target": np.array(sample["target"])
        })

# =========================
# 4. 工具函数：提取 resid_post
# =========================
def collect_hidden_states(data, layer_idx):
    """
    对指定 layer_idx，收集 resid_post hidden states
    返回：
        X: [N, d_model]
        Y: [N, 3]  # 三维：[x(left/right), y(below/above), z(behind/front)]
    """
    X, Y = [], []

    for sample in tqdm(data, desc=f"Layer {layer_idx}", leave=False):
        tokens = model.to_tokens(sample["prompt"], truncate=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_resid_post"
            )

        # 取最后一个 token 的 resid_post
        hidden = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1]
        hidden = hidden.float().cpu().numpy()

        X.append(hidden)
        Y.append(sample["target"])

    return np.stack(X), np.stack(Y)

# =========================
# 5. Layer Sweep Probe
# =========================
print("\n===== Starting Layer Sweep =====")
layer_r2 = []

for layer in range(n_layers):
    print(f"\nProcessing Layer {layer}/{n_layers-1}...")
    
    X_train, Y_train = collect_hidden_states(train_data, layer)
    X_test, Y_test = collect_hidden_states(test_data, layer)

    # Ridge 回归（预测 Δx, Δy, Δz）
    probe = Ridge(alpha=RIDGE_ALPHA)
    probe.fit(X_train, Y_train)

    Y_pred = probe.predict(X_test)

    r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")
    layer_r2.append(r2)

    print(f"Layer {layer:02d} | R² = {r2:.4f}")

# =========================
# 6. 结果输出
# =========================
print("\n" + "="*50)
print("===== Layer Sweep Result =====")
print("="*50)
for i, r2 in enumerate(layer_r2):
    marker = " <<<" if i == np.argmax(layer_r2) else ""
    print(f"Layer {i:02d}: R² = {r2:.4f}{marker}")

best_layer = int(np.argmax(layer_r2))
print("\n" + "="*50)
print(f">>> Best layer: {best_layer} (R²={layer_r2[best_layer]:.4f})")
print("="*50)

# =========================
# 7. 保存结果（可选）
# =========================
results = {
    'layer_r2': layer_r2,
    'best_layer': best_layer,
    'n_layers': n_layers,
    'd_model': d_model,
}

import json
output_file = OUTPUT_FILE_PATH
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_file}")


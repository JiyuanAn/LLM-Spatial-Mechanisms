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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str)
parser.add_argument("--train_data_file_path", "-tr", type=str)
parser.add_argument("--test_data_file_path", "-te", type=str)
parser.add_argument("--output_file_path", "-o", type=str, default=f"probe_spatial_procedure_normalized_{time.strftime('%Y%m%d_%H%M%S')}.json")
parser.add_argument("--ridge_alpha", "-a", type=float, default=100.0, help="Ridge regression alpha (default: 100.0)")
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
RIDGE_ALPHA = args.ridge_alpha  # 使用更大的正则化
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Ridge Alpha: {RIDGE_ALPHA}")

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
# 3. 加载数据 - Spatial Procedure 格式
# =========================
import json
train_data = []
with open(TRAIN_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        train_data.append({
            "prompt": sample["question"],
            "target": np.array(sample["target"]),  # 最终绝对位置坐标
            "start_position": np.array(sample.get("start_position", [0.0, 0.0, 0.0]))
        })

test_data = []
with open(TEST_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        test_data.append({
            "prompt": sample["question"],
            "target": np.array(sample["target"]),  # 最终绝对位置坐标
            "start_position": np.array(sample.get("start_position", [0.0, 0.0, 0.0]))
        })

print(f"Loaded {len(train_data)} training samples")
print(f"Loaded {len(test_data)} test samples")

# 创建标准化器（用训练集拟合）
train_targets = np.stack([s["target"] for s in train_data])
scaler = StandardScaler()
scaler.fit(train_targets)

print(f"\nTarget normalization:")
print(f"  Mean: {scaler.mean_}")
print(f"  Std:  {scaler.scale_}")

# =========================
# 4. 工具函数：提取 resid_post
# =========================
def collect_hidden_states(data, layer_idx):
    """
    对指定 layer_idx，收集 resid_post hidden states
    
    返回：
        X: [N, d_model] - 隐藏状态
        Y: [N, 3]       - 标准化后的最终位置坐标
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

    Y_array = np.stack(Y)
    # 标准化目标值
    Y_normalized = scaler.transform(Y_array)
    
    return np.stack(X), Y_normalized, Y_array  # 返回原始值用于计算真实 R²

# =========================
# 5. Layer Sweep Probe
# =========================
print("\n===== Starting Layer Sweep (with Normalization) =====")
print("Task: Predict final absolute position after executing spatial operations")
print(f"Ridge Alpha: {RIDGE_ALPHA}")
print("="*60)

layer_r2 = []
layer_r2_normalized = []
layer_details = []

for layer in range(n_layers):
    print(f"\nProcessing Layer {layer}/{n_layers-1}...")
    
    X_train, Y_train_norm, Y_train_orig = collect_hidden_states(train_data, layer)
    X_test, Y_test_norm, Y_test_orig = collect_hidden_states(test_data, layer)

    # Ridge 回归（在标准化空间中训练）
    probe = Ridge(alpha=RIDGE_ALPHA)
    probe.fit(X_train, Y_train_norm)

    Y_pred_norm = probe.predict(X_test)
    
    # 反标准化回原始空间
    Y_pred_orig = scaler.inverse_transform(Y_pred_norm)

    # 计算标准化空间的 R²
    r2_norm = r2_score(Y_test_norm, Y_pred_norm, multioutput="uniform_average")
    
    # 计算原始空间的 R²（更有意义）
    r2_orig = r2_score(Y_test_orig, Y_pred_orig, multioutput="uniform_average")
    
    # 计算每个维度的 R²（原始空间）
    r2_x = r2_score(Y_test_orig[:, 0], Y_pred_orig[:, 0])
    r2_y = r2_score(Y_test_orig[:, 1], Y_pred_orig[:, 1])
    r2_z = r2_score(Y_test_orig[:, 2], Y_pred_orig[:, 2])
    
    # 计算 MSE
    mse = mean_squared_error(Y_test_orig, Y_pred_orig)
    
    layer_r2.append(r2_orig)
    layer_r2_normalized.append(r2_norm)
    layer_details.append({
        'layer': layer,
        'r2_original': float(r2_orig),
        'r2_normalized': float(r2_norm),
        'r2_x': float(r2_x),
        'r2_y': float(r2_y),
        'r2_z': float(r2_z),
        'mse': float(mse)
    })

    print(f"Layer {layer:02d} | R²(orig)={r2_orig:7.4f} | R²(norm)={r2_norm:7.4f} | MSE={mse:7.2f}")
    print(f"           Per-dim: x={r2_x:6.4f}, y={r2_y:6.4f}, z={r2_z:6.4f}")

# =========================
# 6. 结果输出
# =========================
print("\n" + "="*70)
print("===== Layer Sweep Result (Spatial Procedure with Normalization) =====")
print("="*70)
for i, detail in enumerate(layer_details):
    marker = " <<<" if i == np.argmax(layer_r2) else ""
    print(f"Layer {i:02d}: R²(orig)={detail['r2_original']:7.4f} | "
          f"R²(norm)={detail['r2_normalized']:7.4f} | MSE={detail['mse']:7.2f}{marker}")

best_layer = int(np.argmax(layer_r2))
print("\n" + "="*70)
print(f">>> Best layer: {best_layer} (R²={layer_r2[best_layer]:.4f})")
best_detail = layer_details[best_layer]
print(f"    Best layer details:")
print(f"    - R² (original space): {best_detail['r2_original']:.4f}")
print(f"    - R² (normalized space): {best_detail['r2_normalized']:.4f}")
print(f"    - R² (x): {best_detail['r2_x']:.4f}")
print(f"    - R² (y): {best_detail['r2_y']:.4f}")
print(f"    - R² (z): {best_detail['r2_z']:.4f}")
print(f"    - MSE: {best_detail['mse']:.2f}")
print("="*70)

# =========================
# 7. 保存结果
# =========================
results = {
    'task': 'spatial_procedure_normalized',
    'model_name': MODEL_NAME,
    'ridge_alpha': RIDGE_ALPHA,
    'normalization': {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()
    },
    'layer_r2': layer_r2,
    'layer_r2_normalized': layer_r2_normalized,
    'layer_details': layer_details,
    'best_layer': best_layer,
    'n_layers': n_layers,
    'd_model': d_model,
    'train_data_file': TRAIN_DATA_FILE_PATH,
    'test_data_file': TEST_DATA_FILE_PATH,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

output_file = OUTPUT_FILE_PATH
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_file}")


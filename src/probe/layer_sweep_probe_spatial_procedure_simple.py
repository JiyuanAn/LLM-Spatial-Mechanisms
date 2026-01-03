"""
简化版 Spatial Procedure Probe：只测试简单移动，不包含旋转/反射
看看模型能否至少学会简单的位移计算
"""
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
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str)
parser.add_argument("--train_data_file_path", "-tr", type=str)
parser.add_argument("--test_data_file_path", "-te", type=str)
parser.add_argument("--output_file_path", "-o", type=str, default=f"probe_spatial_simple_{time.strftime('%Y%m%d_%H%M%S')}.json")
parser.add_argument("--ridge_alpha", "-a", type=float, default=100.0)
args = parser.parse_args()

MODEL_NAME = args.model_name
TRAIN_DATA_FILE_PATH = args.train_data_file_path
TEST_DATA_FILE_PATH = args.test_data_file_path
OUTPUT_FILE_PATH = args.output_file_path
RIDGE_ALPHA = args.ridge_alpha

# =========================
# 配置
# =========================
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = "cuda:0"
DTYPE = torch.float16
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# 加载模型
# =========================
print("Loading model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

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

# =========================
# 加载并过滤数据 - 只保留简单移动
# =========================
def filter_simple_moves(data):
    """只保留只包含移动操作的样本"""
    filtered = []
    for sample in data:
        actions = sample.get('actions', [])
        # 只保留 move 操作
        if all(action['type'] == 'move' for action in actions):
            filtered.append(sample)
    return filtered

with open(TRAIN_DATA_FILE_PATH, "r") as f:
    train_all = json.load(f)
    train_filtered = filter_simple_moves(train_all)

with open(TEST_DATA_FILE_PATH, "r") as f:
    test_all = json.load(f)
    test_filtered = filter_simple_moves(test_all)

train_data = []
for s in train_filtered:
    train_data.append({
        "prompt": s["question"],
        "target": np.array(s["target"])
    })

test_data = []
for s in test_filtered:
    test_data.append({
        "prompt": s["question"],
        "target": np.array(s["target"])
    })

print(f"Original train: {len(train_all)}, filtered: {len(train_data)}")
print(f"Original test: {len(test_all)}, filtered: {len(test_data)}")

if len(train_data) < 50:
    print("⚠️  警告：过滤后训练样本太少！")
    exit(1)

# 标准化
train_targets = np.stack([s["target"] for s in train_data])
scaler = StandardScaler()
scaler.fit(train_targets)

print(f"\nFiltered data statistics:")
print(f"  Mean: {scaler.mean_}")
print(f"  Std:  {scaler.scale_}")

# =========================
# 收集隐藏状态
# =========================
def collect_hidden_states(data, layer_idx):
    X, Y = [], []
    for sample in tqdm(data, desc=f"Layer {layer_idx}", leave=False):
        tokens = model.to_tokens(sample["prompt"], truncate=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_resid_post"
            )
        hidden = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1]
        hidden = hidden.float().cpu().numpy()
        X.append(hidden)
        Y.append(sample["target"])
    
    Y_array = np.stack(Y)
    Y_normalized = scaler.transform(Y_array)
    return np.stack(X), Y_normalized, Y_array

# =========================
# Layer Sweep
# =========================
print("\n===== Layer Sweep (Simple Moves Only) =====")
print(f"Ridge Alpha: {RIDGE_ALPHA}")
print("="*60)

layer_r2 = []
layer_details = []

for layer in range(n_layers):
    X_train, Y_train_norm, Y_train_orig = collect_hidden_states(train_data, layer)
    X_test, Y_test_norm, Y_test_orig = collect_hidden_states(test_data, layer)

    probe = Ridge(alpha=RIDGE_ALPHA)
    probe.fit(X_train, Y_train_norm)
    Y_pred_norm = probe.predict(X_test)
    Y_pred_orig = scaler.inverse_transform(Y_pred_norm)

    r2_orig = r2_score(Y_test_orig, Y_pred_orig, multioutput="uniform_average")
    r2_x = r2_score(Y_test_orig[:, 0], Y_pred_orig[:, 0])
    r2_y = r2_score(Y_test_orig[:, 1], Y_pred_orig[:, 1])
    r2_z = r2_score(Y_test_orig[:, 2], Y_pred_orig[:, 2])
    
    layer_r2.append(r2_orig)
    layer_details.append({
        'layer': layer,
        'r2': float(r2_orig),
        'r2_x': float(r2_x),
        'r2_y': float(r2_y),
        'r2_z': float(r2_z)
    })

    marker = " <<<" if r2_orig == max(layer_r2) else ""
    print(f"Layer {layer:02d} | R²={r2_orig:7.4f} | x:{r2_x:6.4f} y:{r2_y:6.4f} z:{r2_z:6.4f}{marker}")

# =========================
# 结果输出
# =========================
best_layer = int(np.argmax(layer_r2))
print("\n" + "="*60)
print(f">>> Best layer: {best_layer} (R²={layer_r2[best_layer]:.4f})")
print("="*60)

results = {
    'task': 'spatial_procedure_simple_moves_only',
    'model_name': MODEL_NAME,
    'ridge_alpha': RIDGE_ALPHA,
    'num_train_samples': len(train_data),
    'num_test_samples': len(test_data),
    'layer_r2': layer_r2,
    'layer_details': layer_details,
    'best_layer': best_layer,
    'normalization': {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()
    }
}

with open(OUTPUT_FILE_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUTPUT_FILE_PATH}")


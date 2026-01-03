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

# =========================
# 命令行参数解析
# =========================
parser = argparse.ArgumentParser(description='Orientation Reasoning Probe - Layer Sweep')
parser.add_argument("--model_name", "-m", type=str, required=True, 
                    help="Model name (must be in PATHS config)")
parser.add_argument("--train_data_file_path", "-tr", type=str, required=True,
                    help="Path to training data JSON file")
parser.add_argument("--test_data_file_path", "-te", type=str, required=True,
                    help="Path to test data JSON file")
parser.add_argument("--output_file_path", "-o", type=str, 
                    default=f"probe_results_orientation_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    help="Output file path for results")
parser.add_argument("--device", "-d", type=str, default="cuda:0",
                    help="Device to run on (default: cuda:0)")
parser.add_argument("--batch_size", "-b", type=int, default=4,
                    help="Batch size (default: 4)")
parser.add_argument("--ridge_alpha", "-a", type=float, default=1.0,
                    help="Ridge regression alpha (default: 1.0)")
args = parser.parse_args()

MODEL_NAME = args.model_name
TRAIN_DATA_FILE_PATH = args.train_data_file_path
TEST_DATA_FILE_PATH = args.test_data_file_path
OUTPUT_FILE_PATH = args.output_file_path

# =========================
# 1. 基本配置
# =========================
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = args.device
DTYPE = torch.float16
BATCH_SIZE = args.batch_size
RIDGE_ALPHA = args.ridge_alpha
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# 2. 方向编码函数
# =========================
def direction_to_angle_vector(direction: str) -> np.ndarray:
    """
    将方向转换为2D向量（与数据集编码一致）
    
    方向到向量的映射：
    - north: [0, 1]   (正y方向)
    - east: [1, 0]    (正x方向)
    - south: [0, -1]  (负y方向)
    - west: [-1, 0]   (负x方向)
    
    Args:
        direction: 方向字符串 (north/east/south/west)
        
    Returns:
        angle_vector: [x, y]
    """
    direction_vectors = {
        'north': np.array([0, 1], dtype=np.float32),
        'east': np.array([1, 0], dtype=np.float32),
        'south': np.array([0, -1], dtype=np.float32),
        'west': np.array([-1, 0], dtype=np.float32)
    }
    
    if direction.lower() not in direction_vectors:
        raise ValueError(f"Unknown direction: {direction}")
    
    return direction_vectors[direction.lower()]

def angle_vector_to_direction(angle_vector: np.ndarray) -> str:
    """
    将角度向量转换回方向
    
    使用数据集的编码：
    - north: [0, 1]
    - east: [1, 0]
    - south: [0, -1]
    - west: [-1, 0]
    
    Args:
        angle_vector: 2D向量 [x, y]
        
    Returns:
        direction: 最接近的方向字符串
    """
    # 数据集的编码
    reference_vectors = {
        'north': np.array([0, 1], dtype=np.float32),
        'east': np.array([1, 0], dtype=np.float32),
        'south': np.array([0, -1], dtype=np.float32),
        'west': np.array([-1, 0], dtype=np.float32)
    }
    
    # 找到最接近的方向（使用余弦相似度）
    best_direction = 'north'
    max_similarity = -float('inf')
    
    for direction, ref_vec in reference_vectors.items():
        # 计算余弦相似度
        similarity = np.dot(angle_vector, ref_vec) / (np.linalg.norm(angle_vector) * np.linalg.norm(ref_vec) + 1e-8)
        if similarity > max_similarity:
            max_similarity = similarity
            best_direction = direction
    
    return best_direction

# =========================
# 3. 加载模型
# =========================
print("="*60)
print("Orientation Reasoning Probe - Layer Sweep")
print("="*60)
print(f"\nModel: {MODEL_NAME}")
print(f"Model path: {MODEL_PATH}")
print(f"Device: {DEVICE}")
print(f"Train data: {TRAIN_DATA_FILE_PATH}")
print(f"Test data: {TEST_DATA_FILE_PATH}")
print(f"Ridge alpha: {RIDGE_ALPHA}")
print("\nLoading model...")

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

print(f"✓ Loaded model with {n_layers} layers, d_model={d_model}")

# =========================
# 4. 加载数据
# =========================
print("\nLoading data...")

import json

train_data = []
with open(TRAIN_DATA_FILE_PATH, "r", encoding='utf-8') as f:
    data = json.load(f)
    for sample in data:
        # 使用数据中已有的target字段（如果存在），否则重新编码
        if "target" in sample and sample["target"] is not None:
            angle_vector = np.array(sample["target"], dtype=np.float32)
            end_direction = sample.get("end_direction") or sample.get("answer")
        else:
            end_direction = sample.get("end_direction") or sample.get("answer")
            try:
                angle_vector = direction_to_angle_vector(end_direction)
            except ValueError as e:
                print(f"Warning: Skipping sample {sample.get('id', '?')}: {e}")
                continue
        
        train_data.append({
            "prompt": sample["prompt"],
            "target": angle_vector,
            "direction": end_direction,
            "id": sample.get("id", len(train_data))
        })

test_data = []
with open(TEST_DATA_FILE_PATH, "r", encoding='utf-8') as f:
    data = json.load(f)
    for sample in data:
        # 使用数据中已有的target字段（如果存在），否则重新编码
        if "target" in sample and sample["target"] is not None:
            angle_vector = np.array(sample["target"], dtype=np.float32)
            end_direction = sample.get("end_direction") or sample.get("answer")
        else:
            end_direction = sample.get("end_direction") or sample.get("answer")
            try:
                angle_vector = direction_to_angle_vector(end_direction)
            except ValueError as e:
                print(f"Warning: Skipping sample {sample.get('id', '?')}: {e}")
                continue
        
        test_data.append({
            "prompt": sample["prompt"],
            "target": angle_vector,
            "direction": end_direction,
            "id": sample.get("id", len(test_data))
        })

print(f"✓ Loaded {len(train_data)} training samples")
print(f"✓ Loaded {len(test_data)} test samples")

# 显示实际使用的编码
print("\nDirection encoding used in data:")
encoding_examples = {
    'north': np.array([0, 1], dtype=np.float32),
    'east': np.array([1, 0], dtype=np.float32),
    'south': np.array([0, -1], dtype=np.float32),
    'west': np.array([-1, 0], dtype=np.float32)
}
for direction in ['north', 'east', 'south', 'west']:
    vec = encoding_examples[direction]
    print(f"  {direction:6s} → [{vec[0]:6.3f}, {vec[1]:6.3f}]")

# =========================
# 5. 工具函数：提取 resid_post
# =========================
def collect_hidden_states(data, layer_idx):
    """
    对指定 layer_idx，收集 resid_post hidden states
    返回：
        X: [N, d_model]
        Y: [N, 2]  # 二维：[cos(θ), sin(θ)]
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
# 6. Layer Sweep Probe
# =========================
print("\n" + "="*60)
print("Starting Layer Sweep")
print("="*60)

layer_results = []

for layer in range(n_layers):
    print(f"\n[{layer+1}/{n_layers}] Processing Layer {layer}...")
    
    # 收集隐藏状态
    X_train, Y_train = collect_hidden_states(train_data, layer)
    X_test, Y_test = collect_hidden_states(test_data, layer)

    # Ridge 回归（预测 [cos(θ), sin(θ)]）
    probe = Ridge(alpha=RIDGE_ALPHA)
    probe.fit(X_train, Y_train)

    # 预测
    Y_pred = probe.predict(X_test)

    # 计算 R² 分数
    r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")
    
    # 计算分类准确率（将预测的角度向量转换回方向）
    correct = 0
    for i, (pred_vec, true_vec) in enumerate(zip(Y_pred, Y_test)):
        pred_direction = angle_vector_to_direction(pred_vec)
        true_direction = test_data[i]["direction"]
        if pred_direction == true_direction:
            correct += 1
    
    accuracy = correct / len(test_data) if len(test_data) > 0 else 0
    
    # 保存结果
    layer_results.append({
        'layer': layer,
        'r2': float(r2),
        'accuracy': float(accuracy),
        'correct': correct,
        'total': len(test_data)
    })
    
    print(f"  R² = {r2:.4f} | Accuracy = {accuracy:.2%} ({correct}/{len(test_data)})")

# =========================
# 7. 结果输出
# =========================
print("\n" + "="*60)
print("Layer Sweep Results Summary")
print("="*60)

layer_r2 = [res['r2'] for res in layer_results]
layer_acc = [res['accuracy'] for res in layer_results]

best_layer_r2 = int(np.argmax(layer_r2))
best_layer_acc = int(np.argmax(layer_acc))

print("\nLayer | R² Score | Accuracy")
print("-" * 60)
for res in layer_results:
    layer = res['layer']
    r2 = res['r2']
    acc = res['accuracy']
    marker_r2 = " ← Best R²" if layer == best_layer_r2 else ""
    marker_acc = " ← Best Acc" if layer == best_layer_acc else ""
    marker = marker_r2 or marker_acc
    print(f"{layer:5d} | {r2:8.4f} | {acc:7.2%}{marker}")

print("\n" + "="*60)
print(f"Best layer by R²:       Layer {best_layer_r2} (R²={layer_r2[best_layer_r2]:.4f})")
print(f"Best layer by Accuracy: Layer {best_layer_acc} (Acc={layer_acc[best_layer_acc]:.2%})")
print("="*60)

# =========================
# 8. 保存结果
# =========================
results = {
    'model_name': MODEL_NAME,
    'model_path': MODEL_PATH,
    'n_layers': n_layers,
    'd_model': d_model,
    'train_samples': len(train_data),
    'test_samples': len(test_data),
    'ridge_alpha': RIDGE_ALPHA,
    'layer_results': layer_results,
    'best_layer_r2': best_layer_r2,
    'best_layer_accuracy': best_layer_acc,
    'best_r2': float(layer_r2[best_layer_r2]),
    'best_accuracy': float(layer_acc[best_layer_acc]),
    'encoding': '2D directional vector [x, y]',
    'directions': {
        'north': '[0, 1]',
        'east': '[1, 0]',
        'south': '[0, -1]',
        'west': '[-1, 0]'
    }
}

output_file = OUTPUT_FILE_PATH
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Results saved to: {output_file}")
print("\nDone!")


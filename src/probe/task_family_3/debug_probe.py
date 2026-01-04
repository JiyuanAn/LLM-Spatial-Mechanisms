"""
调试版本：添加详细日志，帮助诊断问题
"""
import sys
sys.path.append("./")
sys.path.append("../../")
sys.path.append("../../../")
from config import PATHS

import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str)
parser.add_argument("--train_data_file_path", "-tr", type=str)
parser.add_argument("--test_data_file_path", "-te", type=str)
parser.add_argument("--layer", "-l", type=int, default=0, help="Only process this layer for quick debug")
args = parser.parse_args()

MODEL_NAME = args.model_name
TRAIN_DATA_FILE_PATH = args.train_data_file_path
TEST_DATA_FILE_PATH = args.test_data_file_path

# =========================
# 1. 基本配置
# =========================
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = "cuda:0"
DTYPE = torch.float16
RIDGE_ALPHA = 1.0
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("DEBUG MODE: Layer Sweep Probe with Detailed Logging")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Random Seed: {SEED}")
print(f"Ridge Alpha: {RIDGE_ALPHA}")
print(f"Target Layer: {args.layer}")

# =========================
# 1.5 Prompt 模板
# =========================
INSTRUCTION_TEMPLATE = """You are given a starting position and a sequence of spatial operations.
Each operation transforms the position in 3D space.

{procedure}

Question:
{question}

Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

Instruction:
Output ONLY the letter of the correct option (A, B, C, or D).
Do NOT provide any explanation, reasoning steps, or additional text.
"""

def construct_prompt(sample):
    question_text = sample['question']
    lines = question_text.strip().split('\n')
    
    question_line = ""
    procedure_lines = []
    
    for line in lines:
        if line.strip().startswith("What is"):
            question_line = line.strip()
        elif line.strip():
            procedure_lines.append(line)
    
    procedure = '\n'.join(procedure_lines)
    options = sample['options']
    
    prompt = INSTRUCTION_TEMPLATE.format(
        procedure=procedure,
        question=question_line,
        option_A=options[0],
        option_B=options[1],
        option_C=options[2],
        option_D=options[3]
    )
    
    return prompt

# =========================
# 2. 加载模型
# =========================
print("\nLoading model...")
print(f"Model path: {MODEL_PATH}")

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True
)
hf_model = hf_model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("Converting to HookedTransformer...")

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
# 3. 加载数据
# =========================
import json

print("\n" + "=" * 70)
print("Loading Data...")
print("=" * 70)

train_data = []
with open(TRAIN_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for i, sample in enumerate(data):
        prompt = construct_prompt(sample)
        train_data.append({
            "prompt": prompt,
            "target": np.array(sample["target"], dtype=np.float32)
        })

test_data = []
with open(TEST_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for i, sample in enumerate(data):
        prompt = construct_prompt(sample)
        test_data.append({
            "prompt": prompt,
            "target": np.array(sample["target"], dtype=np.float32)
        })

print(f"✓ Loaded {len(train_data)} training samples")
print(f"✓ Loaded {len(test_data)} test samples")

# 检查数据
print("\nData Sanity Check:")
print(f"  Train sample 0 target: {train_data[0]['target']}")
print(f"  Train sample 0 prompt length: {len(train_data[0]['prompt'])} chars")
print(f"  Test sample 0 target: {test_data[0]['target']}")
print(f"  Test sample 0 prompt length: {len(test_data[0]['prompt'])} chars")

# 统计目标值
train_targets = np.array([s['target'] for s in train_data])
test_targets = np.array([s['target'] for s in test_data])
print(f"\nTarget Statistics:")
print(f"  Train - Mean: {train_targets.mean(axis=0)}, Std: {train_targets.std(axis=0)}")
print(f"  Test  - Mean: {test_targets.mean(axis=0)}, Std: {test_targets.std(axis=0)}")

# =========================
# 4. 提取特征并训练
# =========================
def collect_hidden_states_debug(data, layer_idx):
    X, Y = [], []
    
    print(f"\nCollecting hidden states from layer {layer_idx}...")
    for i, sample in enumerate(tqdm(data[:10], desc=f"Layer {layer_idx} (first 10)")):  # 只处理前10个
        tokens = model.to_tokens(sample["prompt"], truncate=True)
        
        if i == 0:
            print(f"  Sample 0: {tokens.shape[1]} tokens")
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_resid_post"
            )
        
        hidden = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1]
        hidden = hidden.float().cpu().numpy()
        
        if i == 0:
            print(f"  Sample 0 hidden: shape={hidden.shape}, mean={hidden.mean():.4f}, std={hidden.std():.4f}")
            print(f"  Sample 0 hidden: min={hidden.min():.4f}, max={hidden.max():.4f}")
            print(f"  Sample 0 has NaN: {np.isnan(hidden).any()}, has Inf: {np.isinf(hidden).any()}")
        
        X.append(hidden)
        Y.append(sample["target"])
    
    X = np.stack(X)
    Y = np.stack(Y)
    
    print(f"\n  Feature Matrix: shape={X.shape}")
    print(f"  Feature stats: mean={X.mean():.4f}, std={X.std():.4f}")
    print(f"  Feature has NaN: {np.isnan(X).any()}, has Inf: {np.isinf(X).any()}")
    print(f"  Target Matrix: shape={Y.shape}")
    print(f"  Target stats: mean={Y.mean(axis=0)}, std={Y.std(axis=0)}")
    
    return X, Y

print("\n" + "=" * 70)
print(f"Processing Layer {args.layer}...")
print("=" * 70)

X_train, Y_train = collect_hidden_states_debug(train_data, args.layer)
X_test, Y_test = collect_hidden_states_debug(test_data, args.layer)

print(f"\nTraining Ridge Regression (alpha={RIDGE_ALPHA})...")
probe = Ridge(alpha=RIDGE_ALPHA)
probe.fit(X_train, Y_train)

print(f"  Probe coefficients shape: {probe.coef_.shape}")
print(f"  Probe coefficients stats: mean={probe.coef_.mean():.4f}, std={probe.coef_.std():.4f}")
print(f"  Probe intercept: {probe.intercept_}")

Y_pred = probe.predict(X_test)

print(f"\nPredictions:")
print(f"  Shape: {Y_pred.shape}")
print(f"  Stats: mean={Y_pred.mean(axis=0)}, std={Y_pred.std(axis=0)}")
print(f"  First 5 predictions vs ground truth:")
for i in range(min(5, len(Y_pred))):
    print(f"    Pred: {Y_pred[i]}, True: {Y_test[i]}, Error: {np.abs(Y_pred[i] - Y_test[i])}")

# 计算评估指标
r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

print("\n" + "=" * 70)
print("Results:")
print("=" * 70)
print(f"Layer {args.layer:02d} | R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")
print("=" * 70)


import sys
sys.path.append("./")
sys.path.append("../../")
from config import PATHS

import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# =========================
# MLP Probe Model
# =========================
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128], output_dim=3, dropout=0.2):
        super(MLPProbe, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# =========================
# 参数解析
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str)
parser.add_argument("--train_data_file_path", "-tr", type=str)
parser.add_argument("--test_data_file_path", "-te", type=str)
parser.add_argument("--test_layers", type=str, default="0,10,17", help="Comma-separated layer indices to test")
args = parser.parse_args()

MODEL_NAME = args.model_name
TRAIN_DATA_FILE_PATH = args.train_data_file_path
TEST_DATA_FILE_PATH = args.test_data_file_path
TEST_LAYERS = [int(l) for l in args.test_layers.split(',')]

# =========================
# 基本配置
# =========================
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = "cuda:0"
DTYPE = torch.float16
RIDGE_ALPHA = 1.0
SEED = 42

# MLP配置
HIDDEN_DIMS = [128]
DROPOUT = 0.2
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 15

torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*60)
print("COMPARISON: Linear (Ridge) vs MLP Probe")
print("="*60)

# =========================
# 加载模型
# =========================
print("\nLoading model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=DTYPE, trust_remote_code=True
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

model = HookedTransformer.from_pretrained(
    MODEL_NAME, hf_model=hf_model, tokenizer=tokenizer,
    dtype=DTYPE, device=DEVICE,
    fold_ln=False, center_writing_weights=False,
    center_unembed=False, fold_value_biases=False,
)
model.eval()

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
print(f"Model: {n_layers} layers, d_model={d_model}")

# =========================
# 加载数据
# =========================
import json

train_data = []
with open(TRAIN_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        train_data.append({
            "prompt": sample["question"],
            "target": np.array(sample["target"], dtype=np.float32)
        })

test_data = []
with open(TEST_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        test_data.append({
            "prompt": sample["question"],
            "target": np.array(sample["target"], dtype=np.float32)
        })

print(f"Data: {len(train_data)} train, {len(test_data)} test")

# 打印数据统计
train_targets = np.array([s["target"] for s in train_data])
test_targets = np.array([s["target"] for s in test_data])
print(f"\nTarget statistics:")
print(f"  Train mean: {train_targets.mean(axis=0)}")
print(f"  Train std: {train_targets.std(axis=0)}")
print(f"  Test mean: {test_targets.mean(axis=0)}")
print(f"  Test std: {test_targets.std(axis=0)}")

# Baseline R² (predicting mean)
from sklearn.dummy import DummyRegressor
dummy = DummyRegressor(strategy="mean")
dummy.fit(train_targets, train_targets)
dummy_pred = dummy.predict(test_targets)
baseline_r2 = r2_score(test_targets, dummy_pred)
print(f"  Baseline R² (predicting mean): {baseline_r2:.4f}")

# =========================
# 提取隐藏状态
# =========================
def collect_hidden_states(data, layer_idx):
    X, Y = [], []
    for sample in tqdm(data, desc=f"Layer {layer_idx}", leave=False):
        tokens = model.to_tokens(sample["prompt"], truncate=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=f"blocks.{layer_idx}.hook_resid_post"
            )
        hidden = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1]
        hidden = hidden.float().cpu().numpy()
        X.append(hidden)
        Y.append(sample["target"])
    return np.stack(X), np.stack(Y)

# =========================
# 训练函数
# =========================
def train_linear_probe(X_train, Y_train, X_test, Y_test):
    """线性探针 (Ridge)"""
    probe = Ridge(alpha=RIDGE_ALPHA)
    probe.fit(X_train, Y_train)
    Y_pred = probe.predict(X_test)
    r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")
    return r2

def train_linear_probe_with_scaling(X_train, Y_train, X_test, Y_test):
    """线性探针 + 标准化"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    probe = Ridge(alpha=RIDGE_ALPHA)
    probe.fit(X_train_scaled, Y_train)
    Y_pred = probe.predict(X_test_scaled)
    r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")
    return r2

def train_mlp_probe(X_train, Y_train, X_test, Y_test):
    """MLP探针 + 标准化"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train_scaled).to(DEVICE)
    Y_train_t = torch.FloatTensor(Y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test_scaled).to(DEVICE)
    Y_test_t = torch.FloatTensor(Y_test).to(DEVICE)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    probe = MLPProbe(input_dim=X_train.shape[1], hidden_dims=HIDDEN_DIMS, 
                     output_dim=3, dropout=DROPOUT).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(probe.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        probe.train()
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = probe(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
        
        probe.eval()
        with torch.no_grad():
            test_outputs = probe(X_test_t)
            test_loss = criterion(test_outputs, Y_test_t).item()
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_model_state = probe.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            break
    
    if best_model_state is not None:
        probe.load_state_dict(best_model_state)
    
    probe.eval()
    with torch.no_grad():
        Y_pred = probe(X_test_t).cpu().numpy()
    
    r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")
    return r2

# =========================
# 对比测试
# =========================
print("\n" + "="*60)
print("TESTING LAYERS:", TEST_LAYERS)
print("="*60)

results = []

for layer in TEST_LAYERS:
    print(f"\n{'='*60}")
    print(f"LAYER {layer}")
    print(f"{'='*60}")
    
    X_train, Y_train = collect_hidden_states(train_data, layer)
    X_test, Y_test = collect_hidden_states(test_data, layer)
    
    print(f"  Hidden state statistics:")
    print(f"    X_train: mean={X_train.mean():.4f}, std={X_train.std():.4f}, min={X_train.min():.4f}, max={X_train.max():.4f}")
    print(f"    X_test:  mean={X_test.mean():.4f}, std={X_test.std():.4f}, min={X_test.min():.4f}, max={X_test.max():.4f}")
    
    print(f"\n  Training probes...")
    r2_linear = train_linear_probe(X_train, Y_train, X_test, Y_test)
    print(f"    [1] Linear (no scaling):  R² = {r2_linear:.4f}")
    
    r2_linear_scaled = train_linear_probe_with_scaling(X_train, Y_train, X_test, Y_test)
    print(f"    [2] Linear (with scaling): R² = {r2_linear_scaled:.4f}")
    
    r2_mlp = train_mlp_probe(X_train, Y_train, X_test, Y_test)
    print(f"    [3] MLP (with scaling):    R² = {r2_mlp:.4f}")
    
    results.append({
        'layer': layer,
        'linear': r2_linear,
        'linear_scaled': r2_linear_scaled,
        'mlp': r2_mlp
    })

# =========================
# 总结
# =========================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Layer':<10} {'Linear':<15} {'Linear+Scale':<15} {'MLP':<15}")
print("-"*60)
for res in results:
    print(f"{res['layer']:<10} {res['linear']:<15.4f} {res['linear_scaled']:<15.4f} {res['mlp']:<15.4f}")



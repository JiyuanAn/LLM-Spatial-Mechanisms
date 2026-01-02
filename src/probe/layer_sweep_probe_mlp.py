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
from sklearn.metrics import r2_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# =========================
# MLP Probe Model
# =========================
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=3, dropout=0.1):
        """
        MLP探针模型
        Args:
            input_dim: 输入维度（d_model）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（3维空间坐标）
            dropout: dropout比率
        """
        super(MLPProbe, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
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
parser.add_argument("--output_file_path", "-o", type=str, default=f"probe_mlp_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
parser.add_argument("--hidden_dims", type=str, default="512,256", help="Hidden layer dimensions, comma-separated (e.g., '512,256')")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
args = parser.parse_args()

MODEL_NAME = args.model_name
TRAIN_DATA_FILE_PATH = args.train_data_file_path
TEST_DATA_FILE_PATH = args.test_data_file_path
OUTPUT_FILE_PATH = args.output_file_path
HIDDEN_DIMS = [int(d) for d in args.hidden_dims.split(',')]
DROPOUT = args.dropout
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
BATCH_SIZE_TRAIN = args.batch_size
PATIENCE = args.patience

# =========================
# 1. 基本配置
# =========================
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = "cuda:0"
DTYPE = torch.float16
BATCH_SIZE = 4  # for transformer inference
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"MLP Configuration:")
print(f"  Hidden dimensions: {HIDDEN_DIMS}")
print(f"  Dropout: {DROPOUT}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE_TRAIN}")
print(f"  Early stopping patience: {PATIENCE}")

# =========================
# 2. 加载模型
# =========================
print("\nLoading model...")
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
# 3. 加载数据
# =========================
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

print(f"\nLoaded {len(train_data)} training samples and {len(test_data)} test samples")

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
# 5. MLP训练函数
# =========================
def train_mlp_probe(X_train, Y_train, X_test, Y_test, input_dim):
    """
    训练MLP探针
    """
    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    Y_train_tensor = torch.FloatTensor(Y_train).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    Y_test_tensor = torch.FloatTensor(Y_test).to(DEVICE)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE_TRAIN, 
        shuffle=True
    )
    
    # 初始化模型
    probe = MLPProbe(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        output_dim=3,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(probe.parameters(), lr=LEARNING_RATE)
    
    # 早停机制
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 训练循环
    for epoch in range(EPOCHS):
        # 训练阶段
        probe.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = probe(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        probe.eval()
        with torch.no_grad():
            test_outputs = probe(X_test_tensor)
            test_loss = criterion(test_outputs, Y_test_tensor).item()
        
        # 早停检查
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_model_state = probe.state_dict().copy()
        else:
            patience_counter += 1
        
        # 每10个epoch或最后一个epoch打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        
        # 早停
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        probe.load_state_dict(best_model_state)
    
    # 最终预测和评估
    probe.eval()
    with torch.no_grad():
        Y_pred = probe(X_test_tensor).cpu().numpy()
    
    Y_test_np = Y_test_tensor.cpu().numpy()
    r2 = r2_score(Y_test_np, Y_pred, multioutput="uniform_average")
    
    return r2, probe

# =========================
# 6. Layer Sweep Probe
# =========================
print("\n===== Starting Layer Sweep with MLP Probe =====")
layer_r2 = []
best_probes = {}

for layer in range(n_layers):
    print(f"\nProcessing Layer {layer}/{n_layers-1}...")
    
    X_train, Y_train = collect_hidden_states(train_data, layer)
    X_test, Y_test = collect_hidden_states(test_data, layer)
    
    # 训练MLP探针
    r2, probe = train_mlp_probe(X_train, Y_train, X_test, Y_test, d_model)
    layer_r2.append(r2)
    best_probes[layer] = probe
    
    print(f"Layer {layer:02d} | R² = {r2:.4f}")

# =========================
# 7. 结果输出
# =========================
print("\n" + "="*50)
print("===== Layer Sweep Result (MLP Probe) =====")
print("="*50)
for i, r2 in enumerate(layer_r2):
    marker = " <<<" if i == np.argmax(layer_r2) else ""
    print(f"Layer {i:02d}: R² = {r2:.4f}{marker}")

best_layer = int(np.argmax(layer_r2))
print("\n" + "="*50)
print(f">>> Best layer: {best_layer} (R²={layer_r2[best_layer]:.4f})")
print("="*50)

# =========================
# 8. 保存结果
# =========================
results = {
    'layer_r2': layer_r2,
    'best_layer': best_layer,
    'n_layers': n_layers,
    'd_model': d_model,
    'mlp_config': {
        'hidden_dims': HIDDEN_DIMS,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE_TRAIN,
    }
}

output_file = OUTPUT_FILE_PATH
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_file}")

# 保存最佳层的模型权重
best_probe_path = output_file.replace('.json', f'_best_layer_{best_layer}.pt')
torch.save(best_probes[best_layer].state_dict(), best_probe_path)
print(f"Best probe model saved to {best_probe_path}")


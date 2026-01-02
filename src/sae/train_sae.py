"""
SAE Training for Layer 8 Spatial Reasoning Features

科学目标：
- 验证 Layer 8 是否包含稀疏、可分离、与空间状态强相关的内部特征
- 寻找方向/位置选择性 feature
- 寻找状态保持型 feature（空间 working memory）

配置：
- Model: Qwen2.5-7B-Instruct
- Layer: 8 (probe 峰值层)
- Hook: mlp_out (空间/状态特征更稀疏)
- Token: 最后一个 token（与 probe 一致）
- Features: 2048 (第一轮)
"""

import sys
sys.path.append("./")
sys.path.append("../../")
from config import PATHS

import os
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# 导入 SAE 模型
from sae_model import SparseAutoencoder

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Train SAE on Layer 8 MLP activations')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--train_data_file_path", "-tr", type=str, required=True)
parser.add_argument("--test_data_file_path", "-te", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default="./sae_results")
parser.add_argument("--layer", type=int, default=8, help="Target layer (default: 8)")
parser.add_argument("--n_features", type=int, default=2048, help="Number of SAE features")
parser.add_argument("--l1_coeff", type=float, default=1e-3, help="L1 sparsity coefficient")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--max_samples", type=int, default=None, help="Max training samples (None=all)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# =========================
# 基本配置
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
LAYER_IDX = args.layer
N_FEATURES = args.n_features
L1_COEFF = args.l1_coeff
BATCH_SIZE = args.batch_size
LR = args.lr
NUM_EPOCHS = args.num_epochs
SEED = args.seed

torch.manual_seed(SEED)
np.random.seed(SEED)

# 创建输出目录
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = time.strftime('%Y%m%d_%H%M%S')
exp_name = f"L{LAYER_IDX}_F{N_FEATURES}_L1{L1_COEFF}_{timestamp}"
exp_dir = output_dir / exp_name
exp_dir.mkdir(parents=True, exist_ok=True)

print(f"Experiment directory: {exp_dir}")

# =========================
# 加载模型
# =========================
print("="*50)
print("Loading model...")
print(f"Model: {MODEL_NAME}")
print(f"Path: {MODEL_PATH}")
print("="*50)

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

# 获取指定层的 MLP 维度（不同层可能不同）
# 通过运行一个 dummy forward pass 来确定实际维度
dummy_tokens = model.to_tokens("test", truncate=True)
with torch.no_grad():
    _, cache = model.run_with_cache(
        dummy_tokens,
        names_filter=f"blocks.{LAYER_IDX}.hook_mlp_out"
    )
    d_mlp = cache[f"blocks.{LAYER_IDX}.hook_mlp_out"].shape[-1]

print(f"Model loaded: {n_layers} layers, d_model={d_model}")
print(f"Layer {LAYER_IDX} d_mlp={d_mlp}")

# =========================
# 数据加载
# =========================
def load_spatial_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """加载空间推理数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if max_samples is not None:
        data = data[:max_samples]
    
    return data

print("\n" + "="*50)
print("Loading data...")
print("="*50)

train_data_raw = load_spatial_data(args.train_data_file_path, args.max_samples)
test_data_raw = load_spatial_data(args.test_data_file_path, min(1000, len(train_data_raw)//10))

print(f"Train samples: {len(train_data_raw)}")
print(f"Test samples: {len(test_data_raw)}")

# =========================
# 激活收集（关键步骤）
# =========================
def collect_mlp_activations(
    model: HookedTransformer,
    data: List[Dict],
    layer_idx: int,
    hook_point: str = "hook_mlp_out",
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    收集指定 layer 的 MLP 激活
    
    返回：
        activations: [N, d_mlp] tensor
        targets: [N, 3] array (spatial targets)
    """
    activations = []
    targets = []
    
    for sample in tqdm(data, desc=f"Collecting Layer {layer_idx} {hook_point}"):
        prompt = sample["question"]
        target = np.array(sample["target"])
        
        tokens = model.to_tokens(prompt, truncate=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.{hook_point}"
            )
        
        # 取最后一个 token 的激活（与 probe 一致）
        mlp_out = cache[f"blocks.{layer_idx}.{hook_point}"][0, -1]
        
        activations.append(mlp_out.float().cpu())
        targets.append(target)
    
    return torch.stack(activations), np.stack(targets)

print("\n" + "="*50)
print(f"Collecting MLP activations from Layer {LAYER_IDX}...")
print("="*50)

X_train, Y_train = collect_mlp_activations(model, train_data_raw, LAYER_IDX)
X_test, Y_test = collect_mlp_activations(model, test_data_raw, LAYER_IDX)

print(f"Train activations shape: {X_train.shape}")
print(f"Test activations shape: {X_test.shape}")

# 保存激活统计
act_stats = {
    'mean': X_train.mean(dim=0).cpu().numpy().tolist(),
    'std': X_train.std(dim=0).cpu().numpy().tolist(),
    'norm_mean': X_train.norm(dim=-1).mean().item(),
    'norm_std': X_train.norm(dim=-1).std().item(),
}
with open(exp_dir / 'activation_stats.json', 'w') as f:
    json.dump(act_stats, f, indent=2)

# =========================
# 初始化 SAE
# =========================
print("\n" + "="*50)
print("Initializing SAE...")
print("="*50)
print(f"d_in: {d_mlp}")
print(f"n_features: {N_FEATURES}")
print(f"l1_coefficient: {L1_COEFF}")
print(f"expansion_factor: {N_FEATURES / d_mlp:.2f}x")

sae = SparseAutoencoder(
    d_in=d_mlp,
    n_features=N_FEATURES,
    l1_coefficient=L1_COEFF,
    dtype=torch.float32,
).to(DEVICE)

# =========================
# 训练 SAE
# =========================
def train_sae(
    sae: SparseAutoencoder,
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    batch_size: int,
    lr: float,
    num_epochs: int,
) -> Dict:
    """训练 SAE 并返回训练历史"""
    
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    history = {
        'train_loss': [],
        'train_mse': [],
        'train_l1': [],
        'train_l0': [],
        'train_cos_sim': [],
        'test_loss': [],
        'test_mse': [],
        'test_l1': [],
        'test_l0': [],
        'test_cos_sim': [],
    }
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    for epoch in range(num_epochs):
        sae.train()
        
        # Shuffle training data
        perm = torch.randperm(n_samples)
        X_train_shuffled = X_train[perm]
        
        epoch_losses = []
        epoch_stats = {k: [] for k in ['mse', 'l1', 'l0', 'cos_sim']}
        
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i in pbar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch = X_train_shuffled[start_idx:end_idx].to(DEVICE)
            
            optimizer.zero_grad()
            loss_dict = sae.loss(batch)
            loss_dict['total'].backward()
            optimizer.step()
            
            # Normalize decoder (每个 batch 后)
            sae.normalize_decoder()
            
            # 记录
            epoch_losses.append(loss_dict['total'].item())
            for k in ['mse', 'l1', 'l0', 'cos_sim']:
                epoch_stats[k].append(loss_dict[k].item())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'L0': f"{loss_dict['l0'].item():.1f}",
            })
        
        # Epoch 统计
        train_loss = np.mean(epoch_losses)
        train_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
        
        # 测试集评估
        sae.eval()
        with torch.no_grad():
            test_batch_size = 256
            test_losses = []
            test_stats = {k: [] for k in ['mse', 'l1', 'l0', 'cos_sim']}
            
            for i in range(0, X_test.shape[0], test_batch_size):
                batch = X_test[i:i+test_batch_size].to(DEVICE)
                loss_dict = sae.loss(batch)
                test_losses.append(loss_dict['total'].item())
                for k in ['mse', 'l1', 'l0', 'cos_sim']:
                    test_stats[k].append(loss_dict[k].item())
        
        test_loss = np.mean(test_losses)
        test_stats_mean = {k: np.mean(v) for k, v in test_stats.items()}
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        for k in ['mse', 'l1', 'l0', 'cos_sim']:
            history[f'train_{k}'].append(train_stats[k])
            history[f'test_{k}'].append(test_stats_mean[k])
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, MSE: {train_stats['mse']:.4f}, "
              f"L1: {train_stats['l1']:.4f}, L0: {train_stats['l0']:.1f}, "
              f"CosSim: {train_stats['cos_sim']:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f}, MSE: {test_stats_mean['mse']:.4f}, "
              f"L1: {test_stats_mean['l1']:.4f}, L0: {test_stats_mean['l0']:.1f}, "
              f"CosSim: {test_stats_mean['cos_sim']:.4f}")
    
    return history

# 开始训练
history = train_sae(
    sae=sae,
    X_train=X_train,
    X_test=X_test,
    batch_size=BATCH_SIZE,
    lr=LR,
    num_epochs=NUM_EPOCHS,
)

# =========================
# 保存模型和结果
# =========================
print("\n" + "="*50)
print("Saving results...")
print("="*50)

# 保存 SAE 模型
torch.save({
    'model_state_dict': sae.state_dict(),
    'config': {
        'd_in': d_mlp,
        'n_features': N_FEATURES,
        'l1_coefficient': L1_COEFF,
        'layer': LAYER_IDX,
    },
    'history': history,
}, exp_dir / 'sae_checkpoint.pt')

# 保存训练历史
with open(exp_dir / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

# 保存实验配置
config = {
    'model_name': MODEL_NAME,
    'model_path': MODEL_PATH,
    'layer': LAYER_IDX,
    'n_features': N_FEATURES,
    'l1_coefficient': L1_COEFF,
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'num_epochs': NUM_EPOCHS,
    'train_samples': len(train_data_raw),
    'test_samples': len(test_data_raw),
    'd_mlp': d_mlp,
    'timestamp': timestamp,
}
with open(exp_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Results saved to: {exp_dir}")

# =========================
# 可视化训练曲线
# =========================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train')
axes[0, 0].plot(history['test_loss'], label='Test')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Total Loss')
axes[0, 0].set_title('Total Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MSE
axes[0, 1].plot(history['train_mse'], label='Train')
axes[0, 1].plot(history['test_mse'], label='Test')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('Reconstruction MSE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# L1
axes[0, 2].plot(history['train_l1'], label='Train')
axes[0, 2].plot(history['test_l1'], label='Test')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('L1')
axes[0, 2].set_title('L1 Sparsity Loss')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# L0 (sparsity)
axes[1, 0].plot(history['train_l0'], label='Train')
axes[1, 0].plot(history['test_l0'], label='Test')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('L0 (# active features)')
axes[1, 0].set_title('Sparsity (L0)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Cosine Similarity
axes[1, 1].plot(history['train_cos_sim'], label='Train')
axes[1, 1].plot(history['test_cos_sim'], label='Test')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Cosine Similarity')
axes[1, 1].set_title('Reconstruction Similarity')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 1])

# 最终结果总结
final_stats_text = f"""Final Results (Epoch {NUM_EPOCHS}):

Train:
  Loss: {history['train_loss'][-1]:.4f}
  MSE: {history['train_mse'][-1]:.4f}
  L0: {history['train_l0'][-1]:.1f}
  CosSim: {history['train_cos_sim'][-1]:.4f}

Test:
  Loss: {history['test_loss'][-1]:.4f}
  MSE: {history['test_mse'][-1]:.4f}
  L0: {history['test_l0'][-1]:.1f}
  CosSim: {history['test_cos_sim'][-1]:.4f}
"""
axes[1, 2].text(0.1, 0.5, final_stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(exp_dir / 'training_curves.png', dpi=150)
print(f"Plots saved to: {exp_dir / 'training_curves.png'}")

# =========================
# 最终总结
# =========================
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"\nExperiment: {exp_name}")
print(f"Directory: {exp_dir}")
print(f"\nFinal Test Results:")
print(f"  Reconstruction MSE: {history['test_mse'][-1]:.4f}")
print(f"  Sparsity (L0): {history['test_l0'][-1]:.1f} / {N_FEATURES} features")
print(f"  Cosine Similarity: {history['test_cos_sim'][-1]:.4f}")
print(f"\nSparsity: {100 * history['test_l0'][-1] / N_FEATURES:.1f}% features active")
print("\n下一步：运行 analyze_features.py 来分析学到的 features")
print("="*50)


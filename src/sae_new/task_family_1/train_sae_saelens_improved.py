"""
SAE Training using SAELens Library - IMPROVED VERSION
改进版本：解决特征死亡问题

主要改进：
1. 使用所有 token 位置（而非只用最后一个）
2. 更大的 batch size
3. 更好的学习率调度
4. 改进的初始化
"""

import sys
sys.path.append("./")
sys.path.append("../../")
sys.path.append("../../../")
from config import PATHS

import os
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# SAELens imports
try:
    from sae_lens import SAE
    from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
    from sae_lens.saes import TrainingSAE, StandardTrainingSAE, StandardTrainingSAEConfig
    SparseAutoencoder = TrainingSAE
    SAELENS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAELens not installed or import failed: {e}")
    print("Please run: pip install sae-lens")
    SAELENS_AVAILABLE = False
    sys.exit(1)

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Train SAE using SAELens library (Improved Version)')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--train_data_file_path", "-tr", type=str, required=True)
parser.add_argument("--test_data_file_path", "-te", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default="./sae_results")
parser.add_argument("--layer", type=int, help="Target layer")
parser.add_argument("--n_features", type=int, default=8192, help="Number of SAE features (reduced default)")
parser.add_argument("--l1_coeff", type=float, default=1e-6, help="L1 sparsity coefficient (lower default)")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--num_tokens", type=int, default=100000, help="Number of tokens to train on")
parser.add_argument("--max_samples", type=int, default=None, help="Max training samples (None=all)")
parser.add_argument("--use_all_tokens", action='store_true', help="Use all token positions (not just last)")
parser.add_argument("--batch_size", type=int, default=128, help="Training batch size (increased default)")
parser.add_argument("--n_epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# =========================
# 基本配置
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
LAYER_IDX = args.layer
N_FEATURES = args.n_features
L1_COEFF = args.l1_coeff
LR = args.lr
NUM_TOKENS = args.num_tokens
SEED = args.seed
USE_ALL_TOKENS = args.use_all_tokens
BATCH_SIZE = args.batch_size
N_EPOCHS = args.n_epochs

torch.manual_seed(SEED)
np.random.seed(SEED)

# 创建输出目录
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = time.strftime('%Y%m%d_%H%M%S')
exp_name = f"L{LAYER_IDX}_F{N_FEATURES}_L1{L1_COEFF}_{timestamp}_improved"
exp_dir = output_dir / exp_name
exp_dir.mkdir(parents=True, exist_ok=True)

print(f"Experiment directory: {exp_dir}")
print(f"⭐ IMPROVED VERSION - Key changes:")
print(f"  - Use all tokens: {USE_ALL_TOKENS}")
print(f"  - Batch size: {BATCH_SIZE} (vs 32 in original)")
print(f"  - Epochs: {N_EPOCHS} (vs 10 in original)")
print(f"  - Default L1: {L1_COEFF} (vs 1e-5 in original)")
print(f"  - Default features: {N_FEATURES} (vs 16384 in original)")

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

# 获取指定层的 MLP 维度
dummy_tokens = model.to_tokens("test", truncate=True)
with torch.no_grad():
    _, cache = model.run_with_cache(
        dummy_tokens,
        names_filter=f"blocks.{LAYER_IDX}.hook_mlp_out"
    )
    d_mlp = cache[f"blocks.{LAYER_IDX}.hook_mlp_out"].shape[-1]

print(f"Model loaded: {n_layers} layers, d_model={d_model}")
print(f"Layer {LAYER_IDX} d_mlp={d_mlp}")
print(f"Expansion factor: {N_FEATURES / d_mlp:.2f}x")

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
# 改进的激活收集函数
# =========================
def collect_activations_improved(
    model: HookedTransformer,
    data: List[Dict],
    layer_idx: int,
    max_tokens: int = None,
    use_all_tokens: bool = True,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    改进的激活收集函数
    
    改进点：
    1. 可选使用所有 token 位置（增加数据量）
    2. 返回对应的 targets
    3. 更好的进度显示
    """
    activations = []
    targets = []
    total_tokens = 0
    
    pbar = tqdm(data, desc=f"Collecting activations from Layer {layer_idx}")
    
    for sample in pbar:
        if max_tokens and total_tokens >= max_tokens:
            break
            
        prompt = sample["prompt"]
        target = sample.get("target", [0, 0, 0])  # [x, y, z]
        
        tokens = model.to_tokens(prompt, truncate=True)
        seq_len = tokens.shape[1]
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_mlp_out"
            )
        
        mlp_out = cache[f"blocks.{layer_idx}.hook_mlp_out"][0]  # [seq_len, d_mlp]
        
        if use_all_tokens:
            # 使用所有 token 位置
            for pos in range(mlp_out.shape[0]):
                activations.append(mlp_out[pos].float().cpu())
                targets.append(target)
                total_tokens += 1
                
                if max_tokens and total_tokens >= max_tokens:
                    break
        else:
            # 只用最后一个 token（原始方法）
            activations.append(mlp_out[-1].float().cpu())
            targets.append(target)
            total_tokens += seq_len
        
        pbar.set_postfix({'tokens': total_tokens})
    
    return torch.stack(activations), np.array(targets)

# =========================
# 收集数据
# =========================
print("\n" + "="*50)
print("Collecting activations...")
print("="*50)

print("Collecting training activations...")
X_train, Y_train = collect_activations_improved(
    model, train_data_raw, LAYER_IDX, NUM_TOKENS, USE_ALL_TOKENS
)

print("Collecting test activations...")
X_test, Y_test = collect_activations_improved(
    model, test_data_raw, LAYER_IDX, use_all_tokens=USE_ALL_TOKENS
)

print(f"Train activations: {X_train.shape}")
print(f"Test activations: {X_test.shape}")
print(f"Train targets: {Y_train.shape}")
print(f"Test targets: {Y_test.shape}")

# 数据利用率统计
samples_per_feature = X_train.shape[0] / N_FEATURES
print(f"\n📊 Data utilization:")
print(f"  Samples per feature: {samples_per_feature:.2f}")
if samples_per_feature < 0.5:
    print(f"  ⚠️  WARNING: Too few samples per feature! Consider reducing n_features.")
elif samples_per_feature < 2:
    print(f"  ⚠️  Marginal: Consider using --use_all_tokens or reducing n_features.")
else:
    print(f"  ✓ Good ratio")

# 保存激活
torch.save({
    'train': X_train,
    'test': X_test,
    'train_targets': Y_train,
    'test_targets': Y_test,
}, exp_dir / 'activations.pt')

# =========================
# SAE 配置和初始化
# =========================
print("\n" + "="*50)
print("Configuring SAE...")
print("="*50)

hook_name = f"blocks.{LAYER_IDX}.hook_mlp_out"

sae_cfg = StandardTrainingSAEConfig(
    d_in=d_mlp,
    d_sae=N_FEATURES,
    l1_coefficient=L1_COEFF,
    dtype="float32",
    device=str(DEVICE),
    apply_b_dec_to_input=True,
    normalize_activations="none",
)

print(f"SAE Config:")
print(f"  d_in: {d_mlp}")
print(f"  d_sae (n_features): {N_FEATURES}")
print(f"  expansion_factor: {N_FEATURES / d_mlp:.2f}x")
print(f"  l1_coefficient: {L1_COEFF}")

# 创建 SAE
from sae_lens.saes import StandardTrainingSAE
sae = StandardTrainingSAE(sae_cfg)
sae = sae.to(DEVICE)

# =========================
# 训练循环（改进）
# =========================
print("\n" + "="*50)
print("Training SAE...")
print("="*50)

optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))

# 学习率调度器（余弦退火）
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR*0.1)

batch_size = BATCH_SIZE
n_epochs = N_EPOCHS
warm_up_steps = 200  # 增加预热步数

history = {
    'train_loss': [],
    'train_mse': [],
    'train_l1': [],
    'train_l0': [],
    'test_loss': [],
    'test_mse': [],
    'test_l1': [],
    'test_l0': [],
}

print(f"Training settings:")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {n_epochs}")
print(f"  Warm-up steps: {warm_up_steps}")
print(f"  LR scheduler: CosineAnnealingLR")

print("\nStarting training...")
global_step = 0
best_test_loss = float('inf')

for epoch in range(n_epochs):
    sae.train()
    
    # Shuffle
    perm = torch.randperm(X_train.shape[0])
    X_train_shuffled = X_train[perm]
    
    epoch_losses = []
    epoch_stats = {'mse': [], 'l1': [], 'l0': []}
    
    n_batches = (X_train.shape[0] + batch_size - 1) // batch_size
    
    pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{n_epochs}")
    
    for i in pbar:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_train.shape[0])
        batch = X_train_shuffled[start_idx:end_idx].to(DEVICE)
        
        optimizer.zero_grad()
        
        # 学习率预热
        if global_step < warm_up_steps:
            lr_scale = (global_step + 1) / warm_up_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR * lr_scale
        
        # Forward pass
        feature_acts = sae.encode(batch)
        x_recon = sae.decode(feature_acts)
        
        # 损失计算
        mse_loss = torch.nn.functional.mse_loss(x_recon, batch)
        l1_loss = L1_COEFF * feature_acts.abs().sum(dim=-1).mean()
        loss = mse_loss + l1_loss
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        global_step += 1
        
        # 统计
        with torch.no_grad():
            l0 = (feature_acts.abs() > 1e-8).float().sum(dim=-1).mean()
        
        epoch_losses.append(loss.item())
        epoch_stats['mse'].append(mse_loss.item())
        epoch_stats['l1'].append(l1_loss.item())
        epoch_stats['l0'].append(l0.item())
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'L0': f'{l0.item():.0f}',
        })
    
    # 学习率调度
    if global_step >= warm_up_steps:
        scheduler.step()
    
    # Epoch 统计
    train_loss = np.mean(epoch_losses)
    train_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
    
    # 测试集评估
    sae.eval()
    with torch.no_grad():
        test_batch_size = 256
        test_losses = []
        test_stats = {'mse': [], 'l1': [], 'l0': []}
        
        for i in range(0, X_test.shape[0], test_batch_size):
            batch = X_test[i:i+test_batch_size].to(DEVICE)
            
            feature_acts = sae.encode(batch)
            x_recon = sae.decode(feature_acts)
            
            mse = torch.nn.functional.mse_loss(x_recon, batch)
            l1 = L1_COEFF * feature_acts.abs().sum(dim=-1).mean()
            l0 = (feature_acts.abs() > 1e-8).float().sum(dim=-1).mean()
            
            test_losses.append((mse + l1).item())
            test_stats['mse'].append(mse.item())
            test_stats['l1'].append(l1.item())
            test_stats['l0'].append(l0.item())
    
    test_loss = np.mean(test_losses)
    test_stats_mean = {k: np.mean(v) for k, v in test_stats.items()}
    
    # 记录历史
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    for k in ['mse', 'l1', 'l0']:
        history[f'train_{k}'].append(train_stats[k])
        history[f'test_{k}'].append(test_stats_mean[k])
    
    # 打印统计
    print(f"\nEpoch {epoch+1}/{n_epochs}")
    print(f"  Train - Loss: {train_loss:.4f}, MSE: {train_stats['mse']:.4f}, "
          f"L1: {train_stats['l1']:.4f}, L0: {train_stats['l0']:.1f} ({100*train_stats['l0']/N_FEATURES:.1f}%)")
    print(f"  Test  - Loss: {test_loss:.4f}, MSE: {test_stats_mean['mse']:.4f}, "
          f"L1: {test_stats_mean['l1']:.4f}, L0: {test_stats_mean['l0']:.1f} ({100*test_stats_mean['l0']/N_FEATURES:.1f}%)")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 保存最佳模型
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save({
            'model_state_dict': sae.state_dict(),
            'config': {
                'd_in': d_mlp,
                'n_features': N_FEATURES,
                'l1_coefficient': L1_COEFF,
                'layer': LAYER_IDX,
            },
            'epoch': epoch + 1,
            'test_loss': test_loss,
        }, exp_dir / 'sae_checkpoint_best.pt')
        print(f"  💾 Saved best model (test_loss: {test_loss:.4f})")
    
    # 警告检查
    if test_stats_mean['l0'] < N_FEATURES * 0.01:
        print(f"  ⚠️  WARNING: Very sparse! Only {test_stats_mean['l0']:.0f}/{N_FEATURES} features active")
        print(f"      Consider decreasing L1 coefficient")
    elif test_stats_mean['l0'] > N_FEATURES * 0.3:
        print(f"  ⚠️  WARNING: Not sparse enough! {test_stats_mean['l0']:.0f}/{N_FEATURES} features active")
        print(f"      Consider increasing L1 coefficient")

# =========================
# 保存最终结果
# =========================
print("\n" + "="*50)
print("Saving results...")
print("="*50)

# 保存最终 SAE
torch.save({
    'model_state_dict': sae.state_dict(),
    'config': {
        'd_in': d_mlp,
        'n_features': N_FEATURES,
        'l1_coefficient': L1_COEFF,
        'layer': LAYER_IDX,
        'use_all_tokens': USE_ALL_TOKENS,
        'batch_size': BATCH_SIZE,
        'n_epochs': N_EPOCHS,
    },
    'history': history,
}, exp_dir / 'sae_checkpoint.pt')

# 保存训练历史
with open(exp_dir / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

# 保存配置
config_dict = {
    'model_name': MODEL_NAME,
    'layer': LAYER_IDX,
    'n_features': N_FEATURES,
    'l1_coefficient': L1_COEFF,
    'lr': LR,
    'num_tokens': NUM_TOKENS,
    'train_samples': len(train_data_raw),
    'test_samples': len(test_data_raw),
    'train_activations': X_train.shape[0],
    'test_activations': X_test.shape[0],
    'd_mlp': d_mlp,
    'timestamp': timestamp,
    'use_all_tokens': USE_ALL_TOKENS,
    'batch_size': BATCH_SIZE,
    'n_epochs': N_EPOCHS,
    'version': 'improved',
}
with open(exp_dir / 'config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"Results saved to: {exp_dir}")

# =========================
# 最终总结
# =========================
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)

final_l0 = history['test_l0'][-1]
final_l0_pct = 100 * final_l0 / N_FEATURES

print(f"\nExperiment: {exp_name}")
print(f"Directory: {exp_dir}")
print(f"\nFinal Test Results:")
print(f"  MSE: {history['test_mse'][-1]:.4f}")
print(f"  L0 (sparsity): {final_l0:.1f} / {N_FEATURES} features ({final_l0_pct:.1f}%)")
print(f"  Best test loss: {best_test_loss:.4f}")

# 健康检查
print(f"\n📊 Training Health Check:")
if final_l0_pct < 1:
    print(f"  ❌ FAIL: Too sparse ({final_l0_pct:.1f}%)")
    print(f"     → Decrease L1 coefficient (try {L1_COEFF/2:.1e})")
elif final_l0_pct < 5:
    print(f"  ⚠️  WARNING: Very sparse ({final_l0_pct:.1f}%)")
    print(f"     → Consider decreasing L1 (try {L1_COEFF/1.5:.1e})")
elif final_l0_pct > 30:
    print(f"  ⚠️  WARNING: Not sparse enough ({final_l0_pct:.1f}%)")
    print(f"     → Consider increasing L1 (try {L1_COEFF*1.5:.1e})")
else:
    print(f"  ✓ GOOD: Sparsity looks healthy ({final_l0_pct:.1f}%)")

if USE_ALL_TOKENS:
    print(f"  ✓ Using all tokens (increased data utilization)")
else:
    print(f"  ℹ️  Using only last token (try --use_all_tokens for more data)")

print("\n下一步：运行 analyze_features_saelens.py 来分析学到的 features")
print(f"python analyze_features_saelens.py --checkpoint {exp_dir}")
print("="*50)


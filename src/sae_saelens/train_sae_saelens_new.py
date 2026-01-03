"""
SAE Training using SAELens Library
基于 SAELens 官方库的 SAE 训练

科学目标：
- 验证 Layer 8 是否包含稀疏、可分离、与空间状态强相关的内部特征
- 寻找方向/位置选择性 feature
- 使用官方 SAELens 库，更标准、更稳定
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
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# SAELens imports
try:
    from sae_lens import SAE
    from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
    from sae_lens.saes import TrainingSAE, StandardTrainingSAE, StandardTrainingSAEConfig
    # For backward compatibility
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
parser = argparse.ArgumentParser(description='Train SAE using SAELens library')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--train_data_file_path", "-tr", type=str, required=True)
parser.add_argument("--test_data_file_path", "-te", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default="./sae_results")
parser.add_argument("--layer", type=int, default=8, help="Target layer (default: 8)")
parser.add_argument("--n_features", type=int, default=2048, help="Number of SAE features")
parser.add_argument("--l1_coeff", type=float, default=1e-4, help="L1 sparsity coefficient")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--num_tokens", type=int, default=100000, help="Number of tokens to train on")
parser.add_argument("--max_samples", type=int, default=None, help="Max training samples (None=all)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# =========================
# 基本配置
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # SAELens 推荐用 float32
LAYER_IDX = args.layer
N_FEATURES = args.n_features
L1_COEFF = args.l1_coeff
LR = args.lr
NUM_TOKENS = args.num_tokens
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

# 先用 transformers 加载 HF 模型（从本地路径）
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True
)
# 手动将模型移到 GPU
hf_model = hf_model.to(DEVICE)

# 加载 tokenizer（从本地路径）
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
# SAELens 配置
# =========================
print("\n" + "="*50)
print("Configuring SAELens...")
print("="*50)

# 构建 hook point 名称（SAELens 格式）
hook_name = f"blocks.{LAYER_IDX}.hook_mlp_out"

# 创建 SAE 配置
sae_cfg = StandardTrainingSAEConfig(
    d_in=d_mlp,
    d_sae=N_FEATURES,
    l1_coefficient=L1_COEFF,
    dtype="float32",
    device=str(DEVICE),
    apply_b_dec_to_input=True,  # 关键：使用 b_dec 来中心化输入
    normalize_activations="none",  # 不归一化（我们的激活已经是合理尺度）
)

# 创建 Logging 配置（禁用 wandb）
logging_cfg = LoggingConfig(
    log_to_wandb=False,
    wandb_project="spatial-reasoning-sae",
)

# SAELens 训练配置
cfg = LanguageModelSAERunnerConfig(
    # SAE config
    sae=sae_cfg,
    
    # Model config
    model_name=MODEL_NAME,
    hook_name=hook_name,
    hook_eval=hook_name,
    dataset_path="",  # 我们用自定义数据
    is_dataset_tokenized=False,
    
    # Training config
    lr=LR,
    lr_scheduler_name="constant",  # 简单起见用常数学习率
    train_batch_size_tokens=4096,
    context_size=128,
    
    # Training tokens
    training_tokens=NUM_TOKENS,
    
    # Logging
    logger=logging_cfg,
    
    # Checkpointing
    checkpoint_path=str(exp_dir),
    
    # Device
    device=str(DEVICE),
    seed=SEED,
    dtype="float32",
)

print(f"SAE Config:")
print(f"  d_in: {d_mlp}")
print(f"  d_sae (n_features): {N_FEATURES}")
print(f"  expansion_factor: {N_FEATURES / d_mlp:.2f}x")
print(f"  l1_coefficient: {L1_COEFF}")
print(f"  hook_name: {hook_name}")

# =========================
# 自定义数据迭代器
# =========================
class SpatialReasoningDataset:
    """空间推理数据集的迭代器"""
    def __init__(self, data: List[Dict], model: HookedTransformer):
        self.data = data
        self.model = model
        self.index = 0
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        
        sample = self.data[self.index]
        self.index += 1
        
        # 返回 prompt 文本
        return sample["prompt"]
    
    def __len__(self):
        return len(self.data)

# =========================
# 训练 SAE (使用 SAELens)
# =========================
print("\n" + "="*50)
print("Training SAE with SAELens...")
print("="*50)

# 注意：SAELens 的训练循环需要特殊处理
# 这里我们使用更简单的方式：收集激活，然后训练

def collect_activations_for_saelens(
    model: HookedTransformer,
    data: List[Dict],
    layer_idx: int,
    max_tokens: int = None,
) -> torch.Tensor:
    """
    收集激活用于 SAELens 训练
    返回 [n_samples, d_mlp] 的激活矩阵
    """
    activations = []
    total_tokens = 0
    
    for sample in tqdm(data, desc=f"Collecting activations from Layer {layer_idx}"):
        if max_tokens and total_tokens >= max_tokens:
            break
            
        prompt = sample["prompt"]
        tokens = model.to_tokens(prompt, truncate=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_mlp_out"
            )
        
        # 取最后一个 token 的激活
        mlp_out = cache[f"blocks.{layer_idx}.hook_mlp_out"][0, -1]
        activations.append(mlp_out.float().cpu())
        
        total_tokens += tokens.shape[1]
    
    return torch.stack(activations)

# 收集训练和测试激活
print("Collecting training activations...")
X_train = collect_activations_for_saelens(model, train_data_raw, LAYER_IDX, NUM_TOKENS)

print("Collecting test activations...")
X_test = collect_activations_for_saelens(model, test_data_raw, LAYER_IDX)

print(f"Train activations: {X_train.shape}")
print(f"Test activations: {X_test.shape}")

# 保存激活（供后续分析使用）
torch.save({
    'train': X_train,
    'test': X_test,
    'train_targets': np.array([s['target'] for s in train_data_raw[:len(X_train)]]),
    'test_targets': np.array([s['target'] for s in test_data_raw[:len(X_test)]]),
}, exp_dir / 'activations.pt')

# =========================
# 使用 SAELens 训练
# =========================
print("\n" + "="*50)
print("Initializing SAELens SAE...")
print("="*50)

# 创建 SAE
from sae_lens.saes import StandardTrainingSAE
sae = StandardTrainingSAE(sae_cfg)
sae = sae.to(DEVICE)

# 训练循环
optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
batch_size = 32  # 减小 batch size 以提高稳定性
n_epochs = 10  # 增加 epoch 数
warm_up_steps = 100  # 学习率预热

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

print("Starting training...")
global_step = 0
for epoch in range(n_epochs):
    sae.train()
    
    # Shuffle
    perm = torch.randperm(X_train.shape[0])
    X_train_shuffled = X_train[perm]
    
    epoch_losses = []
    epoch_stats = {'mse': [], 'l1': [], 'l0': []}
    
    n_batches = (X_train.shape[0] + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{n_epochs}"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_train.shape[0])
        batch = X_train_shuffled[start_idx:end_idx].to(DEVICE)
        
        optimizer.zero_grad()
        
        # 学习率预热
        if global_step < warm_up_steps:
            lr_scale = (global_step + 1) / warm_up_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR * lr_scale
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
        
        # SAELens forward: encode -> decode
        # 注意：必须用同一个 forward pass 来保持梯度流
        feature_acts = sae.encode(batch)
        x_recon = sae.decode(feature_acts)
        
        # 计算损失
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
            
            # Encode and reconstruct
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
    
    print(f"\nEpoch {epoch+1}/{n_epochs}")
    print(f"  Train - Loss: {train_loss:.4f}, MSE: {train_stats['mse']:.4f}, "
          f"L1: {train_stats['l1']:.4f}, L0: {train_stats['l0']:.1f}")
    print(f"  Test  - Loss: {test_loss:.4f}, MSE: {test_stats_mean['mse']:.4f}, "
          f"L1: {test_stats_mean['l1']:.4f}, L0: {test_stats_mean['l0']:.1f}")

# =========================
# 保存模型和结果
# =========================
print("\n" + "="*50)
print("Saving results...")
print("="*50)

# 保存 SAE
torch.save({
    'model_state_dict': sae.state_dict(),
    'config': {
        'd_in': d_mlp,
        'n_features': N_FEATURES,
        'l1_coefficient': L1_COEFF,
        'layer': LAYER_IDX,
        'saelens_config': cfg.to_dict() if hasattr(cfg, 'to_dict') else str(cfg),
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
    'd_mlp': d_mlp,
    'timestamp': timestamp,
    'saelens_version': 'official',
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
print(f"\nExperiment: {exp_name}")
print(f"Directory: {exp_dir}")
print(f"\nFinal Test Results:")
print(f"  MSE: {history['test_mse'][-1]:.4f}")
print(f"  L0 (sparsity): {history['test_l0'][-1]:.1f} / {N_FEATURES} features")
print(f"\nSparsity: {100 * history['test_l0'][-1] / N_FEATURES:.1f}% features active")
print("\n下一步：运行 analyze_features_saelens.py 来分析学到的 features")
print("="*50)


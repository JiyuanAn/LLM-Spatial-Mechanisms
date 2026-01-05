"""
Supervised SAE Training
监督式 SAE 训练 - 通过添加空间方向监督信号来保留任务相关信息

核心创新：
- 在重构损失基础上添加分类损失
- 确保SAE特征保留空间信息
- 解决无监督SAE丢弃任务信息的问题
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
import torch.nn as nn
import torch.nn.functional as F
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
    SAELENS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAELens not installed or import failed: {e}")
    print("Please run: pip install sae-lens")
    SAELENS_AVAILABLE = False
    sys.exit(1)

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Train Supervised SAE')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--train_data_file_path", "-tr", type=str, required=True)
parser.add_argument("--test_data_file_path", "-te", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default="./sae_results_supervised")
parser.add_argument("--layer", type=int, help="Target layer")
parser.add_argument("--n_features", type=int, default=512, help="Number of SAE features")
parser.add_argument("--l1_coeff", type=float, default=1e-7, help="L1 sparsity coefficient")
parser.add_argument("--task_coeff", type=float, default=0.1, help="Task loss coefficient")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--num_tokens", type=int, default=500000, help="Number of tokens to train on")
parser.add_argument("--max_samples", type=int, default=None, help="Max training samples (None=all)")
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
TASK_COEFF = args.task_coeff  # 新增：任务损失系数
LR = args.lr
NUM_TOKENS = args.num_tokens
SEED = args.seed

torch.manual_seed(SEED)
np.random.seed(SEED)

# 创建输出目录
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = time.strftime('%Y%m%d_%H%M%S')
exp_name = f"L{LAYER_IDX}_F{N_FEATURES}_L1{L1_COEFF}_TASK{TASK_COEFF}_{timestamp}"
exp_dir = output_dir / exp_name
exp_dir.mkdir(parents=True, exist_ok=True)

print(f"Experiment directory: {exp_dir}")
print(f"Supervised SAE with task coefficient: {TASK_COEFF}")

# =========================
# 空间方向标签映射
# =========================
TARGET_TO_IDX = {
    'right': 0, 'left': 1,
    'above': 2, 'below': 3,
    'front': 4, 'behind': 5,
    'unknown': 6  # 处理异常情况
}
IDX_TO_TARGET = {v: k for k, v in TARGET_TO_IDX.items()}
N_CLASSES = 6  # 只计算6个主要类别

print(f"Spatial directions: {list(TARGET_TO_IDX.keys())[:6]}")

# =========================
# 分类器模块
# =========================
class SpatialClassifier(nn.Module):
    """简单的线性分类器，从SAE特征预测空间方向"""
    def __init__(self, n_features: int, n_classes: int = 6):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, n_features] SAE特征
        Returns:
            logits: [batch_size, n_classes] 分类logits
        """
        return self.linear(features)

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

# =========================
# 数据加载（包含标签）
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

# 提取标签（处理可能是列表的情况）
def extract_target(sample):
    """提取目标标签，处理列表和字符串两种情况"""
    target = sample['target']
    if isinstance(target, list):
        return target[0] if len(target) > 0 else 'unknown'
    return target

train_targets = [extract_target(sample) for sample in train_data_raw]
test_targets = [extract_target(sample) for sample in test_data_raw]

# 检查标签分布
from collections import Counter
train_dist = Counter(train_targets)
print(f"\nTrain label distribution:")
for label, count in sorted(train_dist.items()):
    print(f"  {label}: {count} ({100*count/len(train_targets):.1f}%)")

# =========================
# SAELens 配置
# =========================
print("\n" + "="*50)
print("Configuring Supervised SAELens...")
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
print(f"  task_coefficient: {TASK_COEFF}")
print(f"  hook_name: {hook_name}")

# =========================
# 收集激活和标签
# =========================
def collect_activations_with_labels(
    model: HookedTransformer,
    data: List[Dict],
    layer_idx: int,
    max_tokens: int = None,
) -> tuple:
    """
    收集激活和对应的标签
    返回 (activations, labels)
    """
    activations = []
    labels = []
    total_tokens = 0
    
    for sample in tqdm(data, desc=f"Collecting activations from Layer {layer_idx}"):
        if max_tokens and total_tokens >= max_tokens:
            break
            
        prompt = sample["prompt"]
        target = sample["target"]
        # 处理target可能是列表的情况
        if isinstance(target, list):
            target = target[0] if len(target) > 0 else 'unknown'
        
        tokens = model.to_tokens(prompt, truncate=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_mlp_out"
            )
        
        # 取最后一个 token 的激活
        mlp_out = cache[f"blocks.{layer_idx}.hook_mlp_out"][0, -1]
        activations.append(mlp_out.float().cpu())
        
        # 安全地获取标签索引
        label_idx = TARGET_TO_IDX.get(target, TARGET_TO_IDX['unknown'])
        if label_idx == TARGET_TO_IDX['unknown']:
            print(f"\nWarning: Unknown target '{target}', skipping...")
            continue
        labels.append(label_idx)
        
        total_tokens += tokens.shape[1]
    
    return torch.stack(activations), torch.tensor(labels, dtype=torch.long)

# 收集训练和测试数据
print("\nCollecting training activations and labels...")
X_train, y_train = collect_activations_with_labels(model, train_data_raw, LAYER_IDX, NUM_TOKENS)

print("Collecting test activations and labels...")
X_test, y_test = collect_activations_with_labels(model, test_data_raw, LAYER_IDX)

print(f"\nTrain activations: {X_train.shape}, labels: {y_train.shape}")
print(f"Test activations: {X_test.shape}, labels: {y_test.shape}")

# 保存数据
torch.save({
    'train_activations': X_train,
    'train_labels': y_train,
    'test_activations': X_test,
    'test_labels': y_test,
    'target_to_idx': TARGET_TO_IDX,
}, exp_dir / 'activations_with_labels.pt')

# =========================
# 初始化 SAE 和分类器
# =========================
print("\n" + "="*50)
print("Initializing Supervised SAE...")
print("="*50)

from sae_lens.saes import StandardTrainingSAE
sae = StandardTrainingSAE(sae_cfg)
sae = sae.to(DEVICE)

# 初始化分类器
classifier = SpatialClassifier(N_FEATURES, N_CLASSES).to(DEVICE)

print(f"SAE parameters: {sum(p.numel() for p in sae.parameters()):,}")
print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

# =========================
# 训练循环（监督式）
# =========================
# 优化器同时优化SAE和分类器
all_params = list(sae.parameters()) + list(classifier.parameters())
optimizer = torch.optim.Adam(all_params, lr=LR, betas=(0.9, 0.999))

batch_size = 64
n_epochs = 20
warm_up_steps = 200
l1_warmup_steps = 500

history = {
    'train_loss': [],
    'train_mse': [],
    'train_l1': [],
    'train_task': [],  # 新增：任务损失
    'train_l0': [],
    'train_acc': [],   # 新增：训练准确率
    'test_loss': [],
    'test_mse': [],
    'test_l1': [],
    'test_task': [],
    'test_l0': [],
    'test_acc': [],    # 新增：测试准确率
}

print("\n" + "="*50)
print("Starting Supervised Training...")
print("="*50)
print(f"Loss = MSE + {L1_COEFF} * L1 + {TASK_COEFF} * CrossEntropy")
print("="*50)

global_step = 0
best_test_acc = 0.0

for epoch in range(n_epochs):
    sae.train()
    classifier.train()
    
    # Shuffle
    perm = torch.randperm(X_train.shape[0])
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]
    
    epoch_losses = []
    epoch_stats = {'mse': [], 'l1': [], 'task': [], 'l0': [], 'acc': []}
    
    n_batches = (X_train.shape[0] + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{n_epochs}"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_train.shape[0])
        batch_x = X_train_shuffled[start_idx:end_idx].to(DEVICE)
        batch_y = y_train_shuffled[start_idx:end_idx].to(DEVICE)
        
        optimizer.zero_grad()
        
        # 学习率预热
        if global_step < warm_up_steps:
            lr_scale = (global_step + 1) / warm_up_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR * lr_scale
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
        
        # L1 系数预热
        if global_step < l1_warmup_steps:
            l1_scale = (global_step + 1) / l1_warmup_steps
            current_l1_coeff = L1_COEFF * l1_scale
        else:
            current_l1_coeff = L1_COEFF
        
        # SAE forward
        feature_acts = sae.encode(batch_x)
        x_recon = sae.decode(feature_acts)
        
        # 分类器 forward
        logits = classifier(feature_acts)
        
        # 计算损失
        mse_loss = F.mse_loss(x_recon, batch_x)
        l1_loss = current_l1_coeff * feature_acts.abs().sum(dim=-1).mean()
        task_loss = F.cross_entropy(logits, batch_y)
        
        # 总损失（关键：加入任务监督信号）
        loss = mse_loss + l1_loss + TASK_COEFF * task_loss
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        
        optimizer.step()
        
        global_step += 1
        
        # 统计
        with torch.no_grad():
            l0 = (feature_acts.abs() > 1e-8).float().sum(dim=-1).mean()
            pred = logits.argmax(dim=-1)
            acc = (pred == batch_y).float().mean()
        
        epoch_losses.append(loss.item())
        epoch_stats['mse'].append(mse_loss.item())
        epoch_stats['l1'].append(l1_loss.item())
        epoch_stats['task'].append(task_loss.item())
        epoch_stats['l0'].append(l0.item())
        epoch_stats['acc'].append(acc.item())
    
    # Epoch 统计
    train_loss = np.mean(epoch_losses)
    train_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
    
    # 测试集评估
    sae.eval()
    classifier.eval()
    with torch.no_grad():
        test_batch_size = 256
        test_losses = []
        test_stats = {'mse': [], 'l1': [], 'task': [], 'l0': [], 'acc': []}
        
        for i in range(0, X_test.shape[0], test_batch_size):
            batch_x = X_test[i:i+test_batch_size].to(DEVICE)
            batch_y = y_test[i:i+test_batch_size].to(DEVICE)
            
            # Encode and reconstruct
            feature_acts = sae.encode(batch_x)
            x_recon = sae.decode(feature_acts)
            
            # Classify
            logits = classifier(feature_acts)
            
            mse = F.mse_loss(x_recon, batch_x)
            l1 = current_l1_coeff * feature_acts.abs().sum(dim=-1).mean()
            task = F.cross_entropy(logits, batch_y)
            l0 = (feature_acts.abs() > 1e-8).float().sum(dim=-1).mean()
            
            pred = logits.argmax(dim=-1)
            acc = (pred == batch_y).float().mean()
            
            test_losses.append((mse + l1 + TASK_COEFF * task).item())
            test_stats['mse'].append(mse.item())
            test_stats['l1'].append(l1.item())
            test_stats['task'].append(task.item())
            test_stats['l0'].append(l0.item())
            test_stats['acc'].append(acc.item())
    
    test_loss = np.mean(test_losses)
    test_stats_mean = {k: np.mean(v) for k, v in test_stats.items()}
    
    # 记录历史
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    for k in ['mse', 'l1', 'task', 'l0', 'acc']:
        history[f'train_{k}'].append(train_stats[k])
        history[f'test_{k}'].append(test_stats_mean[k])
    
    # 保存最佳模型
    if test_stats_mean['acc'] > best_test_acc:
        best_test_acc = test_stats_mean['acc']
        torch.save({
            'sae_state_dict': sae.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'config': {
                'd_in': d_mlp,
                'n_features': N_FEATURES,
                'l1_coefficient': L1_COEFF,
                'task_coefficient': TASK_COEFF,
                'layer': LAYER_IDX,
                'n_classes': N_CLASSES,
            },
            'epoch': epoch,
            'test_acc': best_test_acc,
        }, exp_dir / 'best_model.pt')
    
    print(f"\nEpoch {epoch+1}/{n_epochs}")
    print(f"  Train - Loss: {train_loss:.4f}, MSE: {train_stats['mse']:.4f}, "
          f"L1: {train_stats['l1']:.4f}, Task: {train_stats['task']:.4f}, "
          f"L0: {train_stats['l0']:.1f}, Acc: {train_stats['acc']:.3f}")
    print(f"  Test  - Loss: {test_loss:.4f}, MSE: {test_stats_mean['mse']:.4f}, "
          f"L1: {test_stats_mean['l1']:.4f}, Task: {test_stats_mean['task']:.4f}, "
          f"L0: {test_stats_mean['l0']:.1f}, Acc: {test_stats_mean['acc']:.3f}")
    
    if test_stats_mean['acc'] == best_test_acc:
        print(f"  ⭐ New best test accuracy!")

# =========================
# 保存最终模型和结果
# =========================
print("\n" + "="*50)
print("Saving results...")
print("="*50)

# 保存最终模型
torch.save({
    'sae_state_dict': sae.state_dict(),
    'classifier_state_dict': classifier.state_dict(),
    'config': {
        'd_in': d_mlp,
        'n_features': N_FEATURES,
        'l1_coefficient': L1_COEFF,
        'task_coefficient': TASK_COEFF,
        'layer': LAYER_IDX,
        'n_classes': N_CLASSES,
        'target_to_idx': TARGET_TO_IDX,
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
    'task_coefficient': TASK_COEFF,
    'lr': LR,
    'num_tokens': NUM_TOKENS,
    'train_samples': len(train_data_raw),
    'test_samples': len(test_data_raw),
    'd_mlp': d_mlp,
    'timestamp': timestamp,
    'method': 'supervised_sae',
    'best_test_acc': best_test_acc,
}
with open(exp_dir / 'config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"Results saved to: {exp_dir}")

# =========================
# 最终总结
# =========================
print("\n" + "="*50)
print("SUPERVISED SAE TRAINING COMPLETE")
print("="*50)
print(f"\nExperiment: {exp_name}")
print(f"Directory: {exp_dir}")
print(f"\nFinal Test Results:")
print(f"  MSE: {history['test_mse'][-1]:.4f}")
print(f"  L0 (sparsity): {history['test_l0'][-1]:.1f} / {N_FEATURES} features")
print(f"  Classification Accuracy: {history['test_acc'][-1]:.3f}")
print(f"  Best Test Accuracy: {best_test_acc:.3f}")
print(f"\nSparsity: {100 * history['test_l0'][-1] / N_FEATURES:.1f}% features active")
print(f"\nComparison:")
print(f"  Expected random chance: {1/N_CLASSES:.3f} (16.7%)")
print(f"  Achieved accuracy: {best_test_acc:.3f} ({100*best_test_acc:.1f}%)")
print("\n下一步：运行 analyze_features_saelens.py 来分析学到的 features")
print("="*50)


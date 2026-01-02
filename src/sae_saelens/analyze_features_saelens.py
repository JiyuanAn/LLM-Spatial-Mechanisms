"""
SAE Feature Analysis using SAELens
基于 SAELens 训练的 SAE 特征分析
"""

import sys
sys.path.append("./")
sys.path.append("../../")
from config import PATHS

import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Analyze SAELens features')
parser.add_argument("--checkpoint", "-c", type=str, required=True,
                    help="Path to SAE checkpoint directory")
parser.add_argument("--output_dir", "-o", type=str, default=None,
                    help="Output directory (default: checkpoint_dir/analysis)")
parser.add_argument("--top_k", type=int, default=50,
                    help="Number of top features to analyze")
args = parser.parse_args()

# =========================
# 加载 checkpoint
# =========================
checkpoint_dir = Path(args.checkpoint)
if args.output_dir is None:
    output_dir = checkpoint_dir / "analysis"
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("="*50)
print("Loading SAELens checkpoint...")
print(f"Checkpoint: {checkpoint_dir}")
print("="*50)

# 加载 SAE
checkpoint_file = checkpoint_dir / 'sae_checkpoint.pt'
checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
sae_config = checkpoint['config']
history = checkpoint['history']

print(f"SAE Config:")
print(f"  d_in: {sae_config['d_in']}")
print(f"  n_features: {sae_config['n_features']}")
print(f"  layer: {sae_config['layer']}")
print(f"  l1_coefficient: {sae_config['l1_coefficient']}")

# 加载激活和 targets
activations_file = checkpoint_dir / 'activations.pt'
if not activations_file.exists():
    print(f"Error: {activations_file} not found!")
    print("Please run train_sae_saelens.py first")
    sys.exit(1)

act_data = torch.load(activations_file, map_location='cpu', weights_only=False)
X_train = act_data['train']
X_test = act_data['test']
Y_train = act_data['train_targets']
Y_test = act_data['test_targets']

print(f"\nData loaded:")
print(f"  Train: {X_train.shape}, Targets: {Y_train.shape}")
print(f"  Test: {X_test.shape}, Targets: {Y_test.shape}")

# =========================
# 加载 SAE 并提取 features
# =========================
print("\n" + "="*50)
print("Loading SAE and extracting features...")
print("="*50)

# 重建 SAE（简化版，只需要 encoder）
try:
    from sae_lens.training.sparse_autoencoder import SparseAutoencoder
    from sae_lens.training.config import LanguageModelSAERunnerConfig
    
    # 从 saved config 重建（简化）
    # 实际使用中，SAELens 的模型加载更复杂，这里我们手动提取 weights
    
    sae_state = checkpoint['model_state_dict']
    
    # 提取 encoder weights
    # 注意：新 API 中 W_enc 形状是 [d_in, d_sae]
    W_enc = sae_state['W_enc']
    b_enc = sae_state['b_enc']
    
    print(f"Encoder weights: {W_enc.shape}")
    print(f"Encoder bias: {b_enc.shape}")
    
except Exception as e:
    print(f"Warning: Could not load full SAELens model: {e}")
    print("Extracting features manually from saved weights...")
    
    sae_state = checkpoint['model_state_dict']
    W_enc = sae_state['W_enc']
    b_enc = sae_state['b_enc']
    b_dec = sae_state.get('b_dec', torch.zeros(sae_config['d_in']))

# 手动计算 features
def encode_features(x: torch.Tensor, W_enc: torch.Tensor, b_enc: torch.Tensor, b_dec: torch.Tensor) -> torch.Tensor:
    """手动编码特征
    在新的 SAELens API 中：
    - W_enc 形状是 [d_in, d_sae]
    - 所以编码是 x @ W_enc + b_enc
    """
    x_centered = x - b_dec
    pre_relu = x_centered @ W_enc + b_enc  # [batch, d_in] @ [d_in, d_sae] = [batch, d_sae]
    return torch.nn.functional.relu(pre_relu)

print("Encoding training features...")
b_dec = sae_state.get('b_dec', torch.zeros(sae_config['d_in']))
train_features = encode_features(X_train, W_enc, b_enc, b_dec)

print("Encoding test features...")
test_features = encode_features(X_test, W_enc, b_enc, b_dec)

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")

# =========================
# Feature 统计分析
# =========================
print("\n" + "="*50)
print("Analyzing feature statistics...")
print("="*50)

feature_stats = {
    'mean': test_features.mean(dim=0).numpy(),
    'std': test_features.std(dim=0).numpy(),
    'max': test_features.max(dim=0)[0].numpy(),
    'activation_freq': (test_features > 0).float().mean(dim=0).numpy(),
}

np.savez(
    output_dir / 'feature_stats.npz',
    **feature_stats
)

print(f"Average activation frequency: {feature_stats['activation_freq'].mean():.4f}")
print(f"Dead features: {(feature_stats['activation_freq'] == 0).sum()}")
print(f"Active features (>1%): {(feature_stats['activation_freq'] > 0.01).sum()}")

# =========================
# Feature-Target 相关性
# =========================
print("\n" + "="*50)
print("Computing feature-target correlations...")
print("="*50)

n_features = sae_config['n_features']
correlations = np.zeros((n_features, 3))
p_values = np.zeros((n_features, 3))

for i in range(n_features):
    feat = test_features[:, i].numpy()
    for j in range(3):
        tgt = Y_test[:, j]
        if feat.std() > 0:  # 避免常数特征
            corr, pval = pearsonr(feat, tgt)
            correlations[i, j] = corr
            p_values[i, j] = pval
        else:
            correlations[i, j] = 0
            p_values[i, j] = 1.0

np.savez(
    output_dir / 'feature_target_correlations.npz',
    correlations=correlations,
    p_values=p_values,
)

# Top features
max_abs_corr = np.abs(correlations).max(axis=1)
best_dim = np.abs(correlations).argmax(axis=1)
top_features = np.argsort(max_abs_corr)[::-1][:args.top_k]

dim_names = ['X(L/R)', 'Y(B/A)', 'Z(B/F)']
print(f"\nTop {args.top_k} features:")
print(f"{'Feature':<10} {'Dim':<10} {'Correlation':<12} {'Act.Freq':<12}")
print("-" * 50)
for feat_idx in top_features[:20]:
    best_d = best_dim[feat_idx]
    corr = correlations[feat_idx, best_d]
    freq = feature_stats['activation_freq'][feat_idx]
    print(f"{feat_idx:<10} {dim_names[best_d]:<10} {corr:>11.4f} {freq:>11.4f}")

# =========================
# 空间维度分类
# =========================
print("\n" + "="*50)
print("Classifying features by spatial dimension...")
print("="*50)

CORR_THRESHOLD = 0.15
FREQ_THRESHOLD = 0.01

dimension_features = {
    'X_positive': [],
    'X_negative': [],
    'Y_positive': [],
    'Y_negative': [],
    'Z_positive': [],
    'Z_negative': [],
}

for i in range(n_features):
    if feature_stats['activation_freq'][i] < FREQ_THRESHOLD:
        continue
    
    for dim in range(3):
        corr = correlations[i, dim]
        if abs(corr) >= CORR_THRESHOLD:
            dim_name = ['X', 'Y', 'Z'][dim]
            direction = 'positive' if corr > 0 else 'negative'
            key = f"{dim_name}_{direction}"
            dimension_features[key].append({
                'feature_idx': i,
                'correlation': float(corr),
                'activation_freq': float(feature_stats['activation_freq'][i]),
            })

for key in dimension_features:
    dimension_features[key].sort(key=lambda x: abs(x['correlation']), reverse=True)

with open(output_dir / 'dimension_features.json', 'w') as f:
    json.dump(dimension_features, f, indent=2)

labels = {
    'X_positive': 'Right',
    'X_negative': 'Left',
    'Y_positive': 'Above',
    'Y_negative': 'Below',
    'Z_positive': 'Front',
    'Z_negative': 'Behind',
}

print("\nSpatial features found:")
for key, label in labels.items():
    count = len(dimension_features[key])
    print(f"  {label:<10}: {count:>3} features")
    if count > 0:
        top = dimension_features[key][0]
        print(f"    Top: Feature {top['feature_idx']}, corr={top['correlation']:.3f}, freq={top['activation_freq']:.3f}")

# 统计稀疏特征
sparse_spatial_features = []
for key in dimension_features:
    sparse_spatial_features.extend([
        f for f in dimension_features[key] 
        if 0.01 < f['activation_freq'] < 0.5
    ])

print(f"\nSparse spatial features (0.01 < freq < 0.5): {len(sparse_spatial_features)}")

# =========================
# Feature Probe
# =========================
print("\n" + "="*50)
print("Training probe on SAE features...")
print("="*50)

X_train_np = train_features.numpy()
X_test_np = test_features.numpy()

probe_full = Ridge(alpha=1.0)
probe_full.fit(X_train_np, Y_train)
Y_pred_full = probe_full.predict(X_test_np)
r2_full = r2_score(Y_test, Y_pred_full, multioutput='uniform_average')

print(f"Probe R² (all {n_features} features): {r2_full:.4f}")

# Top-K probe
for k in [10, 20, 50, 100]:
    if k > len(top_features):
        break
    X_train_top = X_train_np[:, top_features[:k]]
    X_test_top = X_test_np[:, top_features[:k]]
    
    probe_top = Ridge(alpha=1.0)
    probe_top.fit(X_train_top, Y_train)
    Y_pred_top = probe_top.predict(X_test_top)
    r2_top = r2_score(Y_test, Y_pred_top, multioutput='uniform_average')
    
    print(f"Probe R² (top {k} features): {r2_top:.4f}")

probe_results = {
    'r2_full': float(r2_full),
    'r2_top_k': {k: float(r2_score(Y_test, probe_top.fit(X_train_np[:, top_features[:k]], Y_train).predict(X_test_np[:, top_features[:k]]), multioutput='uniform_average'))
                 for k in [10, 20, 50, 100] if k <= len(top_features)}
}
with open(output_dir / 'probe_results.json', 'w') as f:
    json.dump(probe_results, f, indent=2)

# =========================
# 可视化
# =========================
print("\n" + "="*50)
print("Generating visualizations...")
print("="*50)

# Activation frequency distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(feature_stats['activation_freq'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Activation Frequency')
axes[0].set_ylabel('Number of Features')
axes[0].set_title('Feature Activation Frequency Distribution')
axes[0].grid(True, alpha=0.3)

axes[1].hist(np.log10(feature_stats['activation_freq'] + 1e-6), bins=50,
             edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Log10(Activation Frequency)')
axes[1].set_ylabel('Number of Features')
axes[1].set_title('Log-scale Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'activation_frequency.png', dpi=150)
plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
top_30 = top_features[:30]
corr_matrix = correlations[top_30]

sns.heatmap(corr_matrix,
            xticklabels=dim_names,
            yticklabels=top_30,
            cmap='RdBu_r',
            center=0,
            vmin=-0.5, vmax=0.5,
            cbar_kws={'label': 'Pearson Correlation'},
            ax=ax)
ax.set_xlabel('Spatial Dimension')
ax.set_ylabel('Feature Index')
ax.set_title(f'Top {len(top_30)} Features vs Spatial Dimensions')

plt.tight_layout()
plt.savefig(output_dir / 'feature_correlations.png', dpi=150)
plt.close()

print(f"Saved visualizations to {output_dir}")

# =========================
# 最终总结
# =========================
print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print(f"\nResults saved to: {output_dir}")
print("\nKey Findings:")
print(f"  - Total features: {n_features}")
print(f"  - Active features (>1%): {(feature_stats['activation_freq'] > 0.01).sum()}")
print(f"  - Dead features: {(feature_stats['activation_freq'] == 0).sum()}")
print(f"  - Probe R² (all features): {r2_full:.4f}")
print(f"\nSpatial features found:")
for key, label in labels.items():
    count = len(dimension_features[key])
    print(f"  {label:<10}: {count} features")
print(f"\nSparse spatial features: {len(sparse_spatial_features)}")
print("="*50)


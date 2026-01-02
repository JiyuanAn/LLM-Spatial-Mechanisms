"""
SAE Feature Analysis for Spatial Reasoning

分析目标：
1. 找出与空间维度（左/右、上/下、前/后）强相关的 features
2. 识别方向选择性 features
3. 识别状态保持型 features
4. 可视化 feature activations 的分布
"""

import sys
sys.path.append("./")
sys.path.append("../../")
from config import PATHS

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# 导入 SAE 模型
from sae_model import SparseAutoencoder

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Analyze SAE features')
parser.add_argument("--checkpoint", "-c", type=str, required=True, 
                    help="Path to SAE checkpoint")
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--test_data_file_path", "-te", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default=None,
                    help="Output directory (default: same as checkpoint)")
parser.add_argument("--top_k", type=int, default=50,
                    help="Number of top features to analyze")
args = parser.parse_args()

# =========================
# 加载 SAE checkpoint
# =========================
checkpoint_path = Path(args.checkpoint)
if args.output_dir is None:
    output_dir = checkpoint_path.parent / "analysis"
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("="*50)
print("Loading SAE checkpoint...")
print(f"Checkpoint: {checkpoint_path}")
print("="*50)

# PyTorch 2.6+ 需要显式设置 weights_only=False 来加载包含 numpy 的 checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
sae_config = checkpoint['config']
history = checkpoint['history']

print(f"SAE Config:")
print(f"  d_in: {sae_config['d_in']}")
print(f"  n_features: {sae_config['n_features']}")
print(f"  layer: {sae_config['layer']}")
print(f"  l1_coefficient: {sae_config['l1_coefficient']}")

# =========================
# 加载模型
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
LAYER_IDX = sae_config['layer']

print("\n" + "="*50)
print("Loading language model...")
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

# 初始化 SAE
sae = SparseAutoencoder(
    d_in=sae_config['d_in'],
    n_features=sae_config['n_features'],
    l1_coefficient=sae_config['l1_coefficient'],
    dtype=torch.float32,
).to(DEVICE)
sae.load_state_dict(checkpoint['model_state_dict'])
sae.eval()

print("Model loaded successfully!")

# =========================
# 加载测试数据
# =========================
print("\n" + "="*50)
print("Loading test data...")
print("="*50)

with open(args.test_data_file_path, 'r') as f:
    test_data = json.load(f)

print(f"Test samples: {len(test_data)}")

# =========================
# 收集 SAE feature activations
# =========================
def collect_sae_features(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    data: List[Dict],
    layer_idx: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    收集 SAE features 和对应的空间 targets
    
    返回：
        features: [N, n_features]
        targets: [N, 3] (x, y, z)
    """
    all_features = []
    all_targets = []
    
    for sample in tqdm(data, desc="Collecting SAE features"):
        prompt = sample["question"]
        target = np.array(sample["target"])
        
        tokens = model.to_tokens(prompt, truncate=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_mlp_out"
            )
            
            mlp_out = cache[f"blocks.{layer_idx}.hook_mlp_out"][0, -1]
            mlp_out = mlp_out.float().to(DEVICE)
            
            # 通过 SAE encoder 获取 features
            features = sae.encode(mlp_out)
        
        all_features.append(features.cpu())
        all_targets.append(target)
    
    return torch.stack(all_features), np.stack(all_targets)

print("\n" + "="*50)
print("Collecting SAE feature activations...")
print("="*50)

features, targets = collect_sae_features(model, sae, test_data, LAYER_IDX)

print(f"Features shape: {features.shape}")
print(f"Targets shape: {targets.shape}")

# =========================
# 分析 1: Feature 激活统计
# =========================
print("\n" + "="*50)
print("Analyzing feature activation statistics...")
print("="*50)

# 计算每个 feature 的统计量
feature_stats = {
    'mean': features.mean(dim=0).numpy(),
    'std': features.std(dim=0).numpy(),
    'max': features.max(dim=0)[0].numpy(),
    'activation_freq': (features > 0).float().mean(dim=0).numpy(),  # 激活频率
}

# 保存统计量
np.savez(
    output_dir / 'feature_stats.npz',
    mean=feature_stats['mean'],
    std=feature_stats['std'],
    max=feature_stats['max'],
    activation_freq=feature_stats['activation_freq'],
)

print(f"Average activation frequency: {feature_stats['activation_freq'].mean():.4f}")
print(f"Dead features (never activated): {(feature_stats['activation_freq'] == 0).sum()}")
print(f"Highly active features (>10% samples): {(feature_stats['activation_freq'] > 0.1).sum()}")

# 可视化激活频率分布
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(feature_stats['activation_freq'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Activation Frequency')
axes[0].set_ylabel('Number of Features')
axes[0].set_title('Distribution of Feature Activation Frequencies')
axes[0].grid(True, alpha=0.3)

axes[1].hist(np.log10(feature_stats['activation_freq'] + 1e-6), bins=50, 
             edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Log10(Activation Frequency)')
axes[1].set_ylabel('Number of Features')
axes[1].set_title('Log-scale Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'activation_frequency_distribution.png', dpi=150)
print(f"Saved: {output_dir / 'activation_frequency_distribution.png'}")

# =========================
# 分析 2: Feature-Target 相关性
# =========================
print("\n" + "="*50)
print("Computing feature-target correlations...")
print("="*50)

# 计算每个 feature 与三个空间维度的相关系数
# targets: [N, 3] -> [:, 0]=x(left/right), [:, 1]=y(below/above), [:, 2]=z(behind/front)
correlations = np.zeros((sae_config['n_features'], 3))
p_values = np.zeros((sae_config['n_features'], 3))

for i in tqdm(range(sae_config['n_features']), desc="Computing correlations"):
    feat = features[:, i].numpy()
    for j in range(3):
        tgt = targets[:, j]
        corr, pval = pearsonr(feat, tgt)
        correlations[i, j] = corr
        p_values[i, j] = pval

# 保存相关性
np.savez(
    output_dir / 'feature_target_correlations.npz',
    correlations=correlations,
    p_values=p_values,
)

# 计算每个 feature 的最大绝对相关系数
max_abs_corr = np.abs(correlations).max(axis=1)
best_dim = np.abs(correlations).argmax(axis=1)

# 找出 Top-K 最相关的 features
top_k = args.top_k
top_features = np.argsort(max_abs_corr)[::-1][:top_k]

print(f"\nTop {top_k} features by correlation:")
print(f"{'Feature':<10} {'Dim':<10} {'Correlation':<12} {'P-value':<12} {'Act.Freq':<12}")
print("-" * 60)

dim_names = ['X(L/R)', 'Y(B/A)', 'Z(B/F)']
for feat_idx in top_features[:20]:  # 只打印前 20
    best_d = best_dim[feat_idx]
    corr = correlations[feat_idx, best_d]
    pval = p_values[feat_idx, best_d]
    freq = feature_stats['activation_freq'][feat_idx]
    print(f"{feat_idx:<10} {dim_names[best_d]:<10} {corr:>11.4f} {pval:>11.2e} {freq:>11.4f}")

# 保存 top features 信息
top_features_info = {
    'feature_indices': top_features.tolist(),
    'max_abs_correlations': max_abs_corr[top_features].tolist(),
    'best_dimensions': best_dim[top_features].tolist(),
    'correlations': correlations[top_features].tolist(),
    'activation_frequencies': feature_stats['activation_freq'][top_features].tolist(),
}
with open(output_dir / 'top_features.json', 'w') as f:
    json.dump(top_features_info, f, indent=2)

# =========================
# 可视化相关性热图
# =========================
print("\nGenerating correlation heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 热图：Top features 与三个维度的相关性
top_for_heatmap = top_features[:30]
corr_matrix = correlations[top_for_heatmap]

sns.heatmap(corr_matrix, 
            xticklabels=dim_names,
            yticklabels=top_for_heatmap,
            cmap='RdBu_r',
            center=0,
            vmin=-0.5, vmax=0.5,
            cbar_kws={'label': 'Pearson Correlation'},
            ax=axes[0])
axes[0].set_xlabel('Spatial Dimension')
axes[0].set_ylabel('Feature Index')
axes[0].set_title(f'Top {len(top_for_heatmap)} Features vs Spatial Dimensions')

# 散点图：最大相关系数 vs 激活频率
axes[1].scatter(feature_stats['activation_freq'], 
                max_abs_corr, 
                alpha=0.3, s=10)
axes[1].scatter(feature_stats['activation_freq'][top_features[:20]], 
                max_abs_corr[top_features[:20]], 
                color='red', s=50, label=f'Top 20', zorder=5)
axes[1].set_xlabel('Activation Frequency')
axes[1].set_ylabel('Max |Correlation|')
axes[1].set_title('Feature Correlation vs Sparsity')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'feature_correlations.png', dpi=150)
print(f"Saved: {output_dir / 'feature_correlations.png'}")

# =========================
# 分析 3: 按维度分类 features
# =========================
print("\n" + "="*50)
print("Classifying features by spatial dimension...")
print("="*50)

# 定义阈值
CORR_THRESHOLD = 0.15  # 相关系数阈值
FREQ_THRESHOLD = 0.01  # 激活频率阈值（过滤死 features）

dimension_features = {
    'X_positive': [],  # Right
    'X_negative': [],  # Left
    'Y_positive': [],  # Above
    'Y_negative': [],  # Below
    'Z_positive': [],  # Front
    'Z_negative': [],  # Behind
}

for i in range(sae_config['n_features']):
    if feature_stats['activation_freq'][i] < FREQ_THRESHOLD:
        continue  # 跳过死 features
    
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

# 排序并保存
for key in dimension_features:
    dimension_features[key].sort(key=lambda x: abs(x['correlation']), reverse=True)

with open(output_dir / 'dimension_features.json', 'w') as f:
    json.dump(dimension_features, f, indent=2)

print("\nSpatial dimension features:")
labels = {
    'X_positive': 'Right',
    'X_negative': 'Left',
    'Y_positive': 'Above',
    'Y_negative': 'Below',
    'Z_positive': 'Front',
    'Z_negative': 'Behind',
}
for key, label in labels.items():
    count = len(dimension_features[key])
    print(f"  {label:<10}: {count:>3} features")
    if count > 0:
        top_feat = dimension_features[key][0]
        print(f"    Top: Feature {top_feat['feature_idx']}, "
              f"corr={top_feat['correlation']:.3f}, "
              f"freq={top_feat['activation_freq']:.3f}")

# =========================
# 分析 4: Feature Probe（验证 features 的预测能力）
# =========================
print("\n" + "="*50)
print("Training probe on SAE features...")
print("="*50)

# 将 features 作为输入，训练 Ridge 回归预测空间坐标
from sklearn.model_selection import train_test_split

X = features.numpy()
Y = targets

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 全量 features probe
probe_full = Ridge(alpha=1.0)
probe_full.fit(X_train, Y_train)
Y_pred_full = probe_full.predict(X_val)
r2_full = r2_score(Y_val, Y_pred_full, multioutput='uniform_average')

print(f"Probe R² (all {sae_config['n_features']} features): {r2_full:.4f}")

# Top-K features probe
for k in [10, 20, 50, 100]:
    if k > len(top_features):
        break
    X_train_top = X_train[:, top_features[:k]]
    X_val_top = X_val[:, top_features[:k]]
    
    probe_top = Ridge(alpha=1.0)
    probe_top.fit(X_train_top, Y_train)
    Y_pred_top = probe_top.predict(X_val_top)
    r2_top = r2_score(Y_val, Y_pred_top, multioutput='uniform_average')
    
    print(f"Probe R² (top {k} features): {r2_top:.4f}")

# 保存 probe 结果
probe_results = {
    'r2_full': float(r2_full),
    'r2_top_k': {},
}
for k in [10, 20, 50, 100]:
    if k > len(top_features):
        break
    X_train_top = X_train[:, top_features[:k]]
    X_val_top = X_val[:, top_features[:k]]
    probe_top = Ridge(alpha=1.0)
    probe_top.fit(X_train_top, Y_train)
    Y_pred_top = probe_top.predict(X_val_top)
    r2_top = r2_score(Y_val, Y_pred_top, multioutput='uniform_average')
    probe_results['r2_top_k'][k] = float(r2_top)

with open(output_dir / 'probe_results.json', 'w') as f:
    json.dump(probe_results, f, indent=2)

# =========================
# 分析 5: Feature 激活可视化（示例）
# =========================
print("\n" + "="*50)
print("Visualizing example feature activations...")
print("="*50)

# 选择几个最相关的 features 进行可视化
features_to_viz = top_features[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feat_idx in enumerate(features_to_viz):
    ax = axes[idx]
    
    feat_activations = features[:, feat_idx].numpy()
    best_d = best_dim[feat_idx]
    dim_name = dim_names[best_d]
    corr = correlations[feat_idx, best_d]
    
    # 按 target 值着色的散点图
    scatter = ax.scatter(
        range(len(feat_activations)),
        feat_activations,
        c=targets[:, best_d],
        cmap='RdBu_r',
        alpha=0.6,
        s=10,
    )
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Feature Activation')
    ax.set_title(f'Feature {feat_idx}\n{dim_name}, corr={corr:.3f}')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=f'{dim_name} target')

plt.tight_layout()
plt.savefig(output_dir / 'feature_activations_examples.png', dpi=150)
print(f"Saved: {output_dir / 'feature_activations_examples.png'}")

# =========================
# 最终总结
# =========================
print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print(f"\nResults saved to: {output_dir}")
print("\nKey Findings:")
print(f"  - Total features: {sae_config['n_features']}")
print(f"  - Active features (>1% samples): {(feature_stats['activation_freq'] > 0.01).sum()}")
print(f"  - Dead features: {(feature_stats['activation_freq'] == 0).sum()}")
print(f"  - Probe R² (all features): {r2_full:.4f}")
print(f"  - Probe R² (top 50 features): {probe_results['r2_top_k'].get(50, 'N/A')}")
print(f"\nSpatial features found:")
for key, label in labels.items():
    print(f"  {label:<10}: {len(dimension_features[key])} features")
print("\n" + "="*50)

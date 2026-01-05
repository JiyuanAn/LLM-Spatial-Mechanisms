"""
测试特定层的原始激活是否包含空间信息
Test if raw activations contain spatial information
"""

import sys
sys.path.append("./")
sys.path.append("../../")
sys.path.append("../../../")

import torch
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
args = parser.parse_args()

checkpoint_dir = Path(args.checkpoint)

# 加载激活数据
print("="*60)
print("Testing if raw activations contain spatial information")
print("="*60)
print()

act_file = checkpoint_dir / 'activations.pt'
data = torch.load(act_file, map_location='cpu', weights_only=False)

X_train = data['train'].numpy()
X_test = data['test'].numpy()
Y_train = data['train_targets']
Y_test = data['test_targets']

print(f"Data shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  Y_train: {Y_train.shape}")
print(f"  Y_test: {Y_test.shape}")
print()

# 测试原始激活的 probe 性能
print("Training probe on RAW MLP activations...")
probe = Ridge(alpha=1.0)
probe.fit(X_train, Y_train)
Y_pred = probe.predict(X_test)
r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')

print(f"\n{'='*60}")
print(f"Probe R² on RAW activations: {r2:.4f}")
print(f"{'='*60}")

if r2 < 0.1:
    print(f"\n❌ PROBLEM: Layer {checkpoint_dir.name.split('_')[0][1:]} does NOT contain spatial information!")
    print(f"   The probe cannot predict spatial coordinates from raw activations.")
    print(f"\n💡 Solutions:")
    print(f"   1. Try different layers (e.g., 10-16)")
    print(f"   2. Check if you're using the right hook point")
    print(f"   3. Verify the data has spatial labels")
elif r2 < 0.3:
    print(f"\n⚠️  Weak spatial signal (R² = {r2:.4f})")
    print(f"   Layer contains some spatial info but it's weak")
else:
    print(f"\n✓ Good! Layer contains spatial information (R² = {r2:.4f})")
    print(f"  The SAE should be able to extract spatial features")

# 分析每个维度
print(f"\n{'='*60}")
print("Per-dimension analysis:")
print(f"{'='*60}")

dim_names = ['X (Left/Right)', 'Y (Below/Above)', 'Z (Behind/Front)']
for dim in range(3):
    probe_dim = Ridge(alpha=1.0)
    probe_dim.fit(X_train, Y_train[:, dim])
    Y_pred_dim = probe_dim.predict(X_test)
    r2_dim = r2_score(Y_test[:, dim], Y_pred_dim)
    print(f"  {dim_names[dim]:20s}: R² = {r2_dim:6.4f}")

# 检查激活的统计特性
print(f"\n{'='*60}")
print("Activation statistics:")
print(f"{'='*60}")
print(f"  Mean: {X_train.mean():.4f}")
print(f"  Std:  {X_train.std():.4f}")
print(f"  Min:  {X_train.min():.4f}")
print(f"  Max:  {X_train.max():.4f}")

# 计算原始特征与目标的相关性
print(f"\n{'='*60}")
print("Computing feature-target correlations on raw activations...")
print(f"{'='*60}")

d_mlp = X_test.shape[1]
max_corrs = []
for i in range(min(d_mlp, 1000)):  # 只测试前1000个维度
    corrs = []
    for j in range(3):
        if X_test[:, i].std() > 0:
            corr, _ = pearsonr(X_test[:, i], Y_test[:, j])
            corrs.append(abs(corr))
    max_corrs.append(max(corrs))

max_corrs = np.array(max_corrs)
top_10_corrs = np.sort(max_corrs)[-10:][::-1]

print(f"\nTop 10 feature correlations (raw MLP dimensions):")
for i, corr in enumerate(top_10_corrs, 1):
    print(f"  {i:2d}. {corr:.4f}")

print(f"\nMean of top 10: {top_10_corrs.mean():.4f}")

if top_10_corrs[0] < 0.1:
    print(f"\n❌ No individual features are correlated with spatial dimensions")
    print(f"   Maximum correlation: {top_10_corrs[0]:.4f}")
else:
    print(f"\n✓ Some features show correlation with spatial dimensions")

print(f"\n{'='*60}")
print("Summary:")
print(f"{'='*60}")
print(f"1. Probe R² on raw activations: {r2:.4f}")
print(f"2. Max feature correlation: {top_10_corrs[0]:.4f}")

if r2 > 0.3 and top_10_corrs[0] > 0.15:
    print(f"\n✓ This layer is good for spatial features!")
    print(f"  The SAE training issue may be due to:")
    print(f"  - L1 coefficient needs tuning")
    print(f"  - Using all tokens dilutes signal (try last token only)")
    print(f"  - Need more training epochs")
elif r2 > 0.1:
    print(f"\n⚠️  Weak but present signal")
    print(f"  Try adjusting SAE hyperparameters")
else:
    print(f"\n❌ This layer does NOT contain usable spatial information")
    print(f"  Try a different layer!")

print()


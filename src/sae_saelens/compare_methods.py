"""
比较 Gradient-Based 和 Ablation 方法的结果

对比维度:
1. 特征重要性排名的一致性
2. Spatial features 的识别效果
3. 计算效率
4. 预测准确性
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import spearmanr, kendalltau

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Compare Gradient and Ablation Methods')
parser.add_argument("--gradient_results", "-g", type=str, required=True,
                    help="Path to gradient attribution results JSON")
parser.add_argument("--ablation_results", "-a", type=str, required=True,
                    help="Path to ablation results JSON")
parser.add_argument("--output_dir", "-o", type=str, default=None)
args = parser.parse_args()

# =========================
# 加载结果
# =========================
print("="*60)
print("Loading results...")
print("="*60)

with open(args.gradient_results, 'r') as f:
    gradient_data = json.load(f)

with open(args.ablation_results, 'r') as f:
    ablation_data = json.load(f)

if args.output_dir is None:
    output_dir = Path(args.gradient_results).parent / "method_comparison"
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Gradient method: {gradient_data.get('method', 'N/A')}")
print(f"Ablation baseline accuracy: {ablation_data['results']['baseline']['accuracy']:.4f}")
print(f"Gradient accuracy: {gradient_data['attribution_results']['accuracy']:.4f}")

# =========================
# 1. 特征重要性排名对比
# =========================
print("\n" + "="*60)
print("Comparing Feature Importance Rankings")
print("="*60)

# Extract feature importance from gradient method
gradient_importance = gradient_data['attribution_results']['average_attributions']
gradient_ranking = [(i, score) for i, score in enumerate(gradient_importance)]
gradient_ranking.sort(key=lambda x: x[1], reverse=True)

# Extract feature importance from ablation method
# For ablation, we can use the performance drop when ablating each feature
# Here we approximate using spatial features' effect
spatial_feature_ids = ablation_data.get('spatial_feature_ids', [])
random_feature_ids = ablation_data.get('random_feature_ids', [])

# Create ablation-based importance scores
ablation_importance = np.zeros(len(gradient_importance))

# Spatial features get high scores (based on their ablation effect)
spatial_drop = ablation_data['results']['spatial_ablation']['drop']
random_drop = ablation_data['results']['random_ablation']['drop']

# Simple heuristic: spatial features get spatial_drop score, others get random_drop score
for feat_id in spatial_feature_ids:
    if feat_id < len(ablation_importance):
        ablation_importance[feat_id] = spatial_drop / len(spatial_feature_ids)

# Note: This is a simplified comparison. Full per-feature ablation would be expensive.

# Compute ranking correlation for spatial features only
gradient_spatial_scores = [gradient_importance[i] for i in spatial_feature_ids 
                          if i < len(gradient_importance)]
ablation_spatial_scores = [ablation_importance[i] for i in spatial_feature_ids 
                          if i < len(ablation_importance)]

if len(gradient_spatial_scores) > 1 and len(ablation_spatial_scores) > 1:
    # Rank correlation
    spearman_corr, spearman_p = spearmanr(gradient_spatial_scores, ablation_spatial_scores)
    print(f"\nSpearman correlation (spatial features): {spearman_corr:.4f} (p={spearman_p:.4f})")
else:
    spearman_corr = None
    print("\nInsufficient data for correlation analysis")

# =========================
# 2. Spatial Features 识别效果对比
# =========================
print("\n" + "="*60)
print("Spatial Features Detection Comparison")
print("="*60)

# Gradient method: top-k features by attribution
n_spatial = len(spatial_feature_ids)
gradient_top_k = [f[0] for f in gradient_ranking[:n_spatial]]
gradient_overlap = len(set(gradient_top_k) & set(spatial_feature_ids))
gradient_overlap_pct = 100 * gradient_overlap / n_spatial if n_spatial > 0 else 0

# Ablation method: directly uses identified spatial features
ablation_overlap = n_spatial  # 100% by definition
ablation_overlap_pct = 100.0

print(f"\nSpatial features: {n_spatial}")
print(f"Gradient method top-{n_spatial} overlap: {gradient_overlap} ({gradient_overlap_pct:.1f}%)")
print(f"Ablation method: {ablation_overlap} (100% by definition)")

# =========================
# 3. 效果对比
# =========================
print("\n" + "="*60)
print("Effectiveness Comparison")
print("="*60)

# Gradient: spatial vs non-spatial attribution ratio
gradient_spatial_comp = gradient_data.get('spatial_comparison', {})
gradient_ratio = gradient_spatial_comp.get('spatial_vs_nonspatial_ratio', 0)

# Ablation: spatial vs random drop ratio
baseline_acc = ablation_data['results']['baseline']['accuracy']
spatial_drop = ablation_data['results']['spatial_ablation']['drop']
random_drop = ablation_data['results']['random_ablation']['drop']
ablation_ratio = spatial_drop / random_drop if random_drop > 0 else float('inf')

print(f"\nGradient method:")
print(f"  Spatial/Non-spatial attribution ratio: {gradient_ratio:.2f}x")
print(f"\nAblation method:")
print(f"  Spatial/Random drop ratio: {ablation_ratio:.2f}x")
print(f"  Spatial drop: {spatial_drop:.4f} ({spatial_drop/baseline_acc*100:.1f}% of baseline)")
print(f"  Random drop: {random_drop:.4f} ({random_drop/baseline_acc*100:.1f}% of baseline)")

# =========================
# 可视化对比
# =========================
print("\n" + "="*60)
print("Generating comparison visualizations...")
print("="*60)

# Figure 1: Method Comparison Overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Overlap comparison
ax1 = axes[0, 0]
methods = ['Gradient\nTop-K', 'Ablation\nSpatial']
overlaps = [gradient_overlap_pct, ablation_overlap_pct]
colors = ['skyblue', 'lightcoral']
bars = ax1.bar(methods, overlaps, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Overlap with Spatial Features (%)')
ax1.set_title('Spatial Feature Detection')
ax1.set_ylim([0, 105])
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, overlaps):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# Subplot 2: Effect size comparison
ax2 = axes[0, 1]
methods = ['Gradient\nRatio', 'Ablation\nRatio']
ratios = [gradient_ratio, ablation_ratio]
colors = ['skyblue', 'lightcoral']
bars = ax2.bar(methods, ratios, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Spatial vs Non-Spatial/Random Ratio')
ax2.set_title('Effect Size Comparison')
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, ratios):
    height = bar.get_height()
    if np.isfinite(height):
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x', ha='center', va='bottom', fontweight='bold')

# Subplot 3: Top features comparison
ax3 = axes[1, 0]
top_n = 30
gradient_top_features = [f[0] for f in gradient_ranking[:top_n]]
gradient_top_scores = [f[1] for f in gradient_ranking[:top_n]]
is_spatial = [1 if f in spatial_feature_ids else 0 for f in gradient_top_features]

x = np.arange(top_n)
colors_top = ['red' if s else 'blue' for s in is_spatial]
bars = ax3.bar(x, gradient_top_scores, color=colors_top, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Feature Rank')
ax3.set_ylabel('Attribution Score (Gradient)')
ax3.set_title(f'Top {top_n} Features by Gradient Attribution')
ax3.grid(True, alpha=0.3, axis='y')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Spatial'),
    Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='Non-Spatial')
]
ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Subplot 4: Accuracy comparison
ax4 = axes[1, 1]
methods = ['Gradient\nBaseline', 'Ablation\nBaseline', 'Ablation\nSpatial Ablated', 'Ablation\nRandom Ablated']
accuracies = [
    gradient_data['attribution_results']['accuracy'],
    ablation_data['results']['baseline']['accuracy'],
    ablation_data['results']['spatial_ablation']['accuracy'],
    ablation_data['results']['random_ablation']['accuracy']
]
colors_acc = ['skyblue', 'lightcoral', 'red', 'green']
bars = ax4.bar(methods, accuracies, color=colors_acc, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Accuracy')
ax4.set_title('Model Performance Comparison')
ax4.set_ylim([0, 1.0])
ax4.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, accuracies):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'method_comparison_overview.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: method_comparison_overview.png")
plt.close()

# Figure 2: Scatter plot of feature importance (if we had per-feature ablation data)
# This is a placeholder for now since we don't have per-feature ablation
# In a full implementation, you would run individual ablations for each feature

# =========================
# 生成对比报告
# =========================
print("\n" + "="*60)
print("Generating comparison report...")
print("="*60)

report_file = output_dir / 'method_comparison_report.txt'
with open(report_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("Gradient vs Ablation Method Comparison\n")
    f.write("="*60 + "\n\n")
    
    f.write("1. OVERVIEW\n")
    f.write("-"*60 + "\n")
    f.write(f"Gradient Method: {gradient_data.get('method', 'N/A')}\n")
    f.write(f"Number of features: {len(gradient_importance)}\n")
    f.write(f"Number of spatial features: {n_spatial}\n\n")
    
    f.write("2. ACCURACY COMPARISON\n")
    f.write("-"*60 + "\n")
    f.write(f"Gradient baseline accuracy: {gradient_data['attribution_results']['accuracy']:.4f}\n")
    f.write(f"Ablation baseline accuracy: {ablation_data['results']['baseline']['accuracy']:.4f}\n")
    f.write(f"Ablation after spatial ablation: {ablation_data['results']['spatial_ablation']['accuracy']:.4f}\n")
    f.write(f"Ablation after random ablation: {ablation_data['results']['random_ablation']['accuracy']:.4f}\n\n")
    
    f.write("3. SPATIAL FEATURE DETECTION\n")
    f.write("-"*60 + "\n")
    f.write(f"Spatial features identified: {n_spatial}\n")
    f.write(f"Gradient method top-{n_spatial} overlap: {gradient_overlap} ({gradient_overlap_pct:.1f}%)\n")
    f.write(f"Ablation method: {ablation_overlap} (100% by definition)\n\n")
    
    f.write("4. EFFECT SIZE\n")
    f.write("-"*60 + "\n")
    f.write(f"Gradient spatial/non-spatial ratio: {gradient_ratio:.2f}x\n")
    f.write(f"Ablation spatial/random drop ratio: {ablation_ratio:.2f}x\n\n")
    
    f.write("5. KEY FINDINGS\n")
    f.write("-"*60 + "\n")
    
    if gradient_overlap_pct > 70:
        f.write("✓ Gradient method shows STRONG agreement with spatial features\n")
        f.write(f"  ({gradient_overlap_pct:.1f}% overlap with identified spatial features)\n\n")
    elif gradient_overlap_pct > 40:
        f.write("~ Gradient method shows MODERATE agreement with spatial features\n")
        f.write(f"  ({gradient_overlap_pct:.1f}% overlap with identified spatial features)\n\n")
    else:
        f.write("✗ Gradient method shows WEAK agreement with spatial features\n")
        f.write(f"  ({gradient_overlap_pct:.1f}% overlap with identified spatial features)\n\n")
    
    if gradient_ratio > 1.5:
        f.write("✓ Gradient method detects STRONG differential importance\n")
        f.write(f"  (spatial features have {gradient_ratio:.2f}x higher attribution)\n\n")
    elif gradient_ratio > 1.2:
        f.write("~ Gradient method detects MODERATE differential importance\n")
        f.write(f"  (spatial features have {gradient_ratio:.2f}x higher attribution)\n\n")
    else:
        f.write("✗ Gradient method detects WEAK differential importance\n")
        f.write(f"  (spatial features have {gradient_ratio:.2f}x higher attribution)\n\n")
    
    if ablation_ratio > 2.0:
        f.write("✓ Ablation method shows STRONG causal evidence\n")
        f.write(f"  (spatial ablation causes {ablation_ratio:.2f}x more damage than random)\n\n")
    elif ablation_ratio > 1.5:
        f.write("~ Ablation method shows MODERATE causal evidence\n")
        f.write(f"  (spatial ablation causes {ablation_ratio:.2f}x more damage than random)\n\n")
    else:
        f.write("✗ Ablation method shows WEAK causal evidence\n")
        f.write(f"  (spatial ablation causes {ablation_ratio:.2f}x more damage than random)\n\n")
    
    f.write("6. ADVANTAGES & DISADVANTAGES\n")
    f.write("-"*60 + "\n")
    f.write("Gradient Method:\n")
    f.write("  ✓ Much faster (no need to re-run model for each feature)\n")
    f.write("  ✓ Provides continuous importance scores\n")
    f.write("  ✓ Can analyze all features simultaneously\n")
    f.write("  ✗ Attribution is correlation, not causation\n")
    f.write("  ✗ Sensitive to gradient noise\n\n")
    
    f.write("Ablation Method:\n")
    f.write("  ✓ Provides causal evidence (intervention-based)\n")
    f.write("  ✓ Direct measurement of feature impact\n")
    f.write("  ✗ Very slow (requires re-running model many times)\n")
    f.write("  ✗ Binary effect (ablate or not)\n")
    f.write("  ✗ Doesn't scale to per-feature analysis\n\n")
    
    f.write("7. RECOMMENDATIONS\n")
    f.write("-"*60 + "\n")
    f.write("Use Gradient Method when:\n")
    f.write("  - You need fast feature importance screening\n")
    f.write("  - You want to analyze many features\n")
    f.write("  - You need fine-grained attribution scores\n\n")
    
    f.write("Use Ablation Method when:\n")
    f.write("  - You need causal evidence\n")
    f.write("  - You want to validate specific features\n")
    f.write("  - You need interpretable intervention effects\n\n")
    
    f.write("Best Practice:\n")
    f.write("  1. Use Gradient method for initial screening\n")
    f.write("  2. Use Ablation method to validate top candidates\n")
    f.write("  3. Combine both for comprehensive analysis\n")
    f.write("="*60 + "\n")

print(f"✓ Report saved to: {report_file}")

# =========================
# 保存对比结果
# =========================
comparison_results = {
    'gradient': {
        'method': gradient_data.get('method', 'N/A'),
        'accuracy': gradient_data['attribution_results']['accuracy'],
        'spatial_ratio': gradient_ratio,
        'top_k_overlap': gradient_overlap,
        'top_k_overlap_pct': gradient_overlap_pct,
    },
    'ablation': {
        'baseline_accuracy': ablation_data['results']['baseline']['accuracy'],
        'spatial_ablation_accuracy': ablation_data['results']['spatial_ablation']['accuracy'],
        'random_ablation_accuracy': ablation_data['results']['random_ablation']['accuracy'],
        'spatial_drop': spatial_drop,
        'random_drop': random_drop,
        'drop_ratio': ablation_ratio,
    },
    'comparison': {
        'spatial_feature_count': n_spatial,
        'gradient_top_k_overlap_pct': gradient_overlap_pct,
        'spearman_correlation': float(spearman_corr) if spearman_corr is not None else None,
    }
}

results_file = output_dir / 'comparison_results.json'
with open(results_file, 'w') as f:
    json.dump(comparison_results, f, indent=2)
print(f"✓ Comparison results saved to: {results_file}")

# =========================
# 总结
# =========================
print("\n" + "="*60)
print("METHOD COMPARISON COMPLETE")
print("="*60)
print(f"\nResults saved to: {output_dir}/")
print("\nKey Findings:")
print(f"  - Gradient method top-{n_spatial} overlap: {gradient_overlap_pct:.1f}%")
print(f"  - Gradient spatial/non-spatial ratio: {gradient_ratio:.2f}x")
print(f"  - Ablation spatial/random ratio: {ablation_ratio:.2f}x")

if gradient_overlap_pct > 70 and gradient_ratio > 1.5:
    print("\n✓ STRONG agreement between methods!")
    print("  Both methods identify spatial features as important.")
elif gradient_overlap_pct > 40:
    print("\n~ MODERATE agreement between methods")
    print("  Some consistency, but not perfect.")
else:
    print("\n✗ WEAK agreement between methods")
    print("  Methods identify different features as important.")

print("\nRecommendation:")
print("  - Use gradient method for fast screening")
print("  - Use ablation method for causal validation")
print("  - Combine both for best results")
print("="*60)



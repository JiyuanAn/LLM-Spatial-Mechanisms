"""
可视化 Gradient-Based Feature Attribution 结果
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Visualize Gradient Attribution Results')
parser.add_argument("--results_file", "-r", type=str, required=True)
parser.add_argument("--sae_checkpoint", "-c", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default=None)
args = parser.parse_args()

# =========================
# 加载结果
# =========================
print("="*60)
print("Loading results...")
print("="*60)

with open(args.results_file, 'r') as f:
    results = json.load(f)

checkpoint_dir = Path(args.sae_checkpoint)
if args.output_dir is None:
    output_dir = checkpoint_dir / "gradient_attribution"
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

method = results['method']
avg_attributions = np.array(results['attribution_results']['average_attributions'])
feature_ranking = results['attribution_results']['feature_importance_ranking']
spatial_comparison = results.get('spatial_comparison', {})

print(f"Method: {method}")
print(f"Features: {len(avg_attributions)}")
print(f"Accuracy: {results['attribution_results']['accuracy']:.4f}")

# =========================
# 图 1: Feature Attribution Distribution
# =========================
print("\nGenerating attribution distribution plot...")

plt.figure(figsize=(12, 5))

# Subplot 1: Histogram
plt.subplot(1, 2, 1)
plt.hist(avg_attributions, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Attribution Score')
plt.ylabel('Number of Features')
plt.title(f'Feature Attribution Distribution\n({method})')
plt.grid(True, alpha=0.3)

# Subplot 2: Sorted attributions
plt.subplot(1, 2, 2)
sorted_attrs = np.sort(avg_attributions)[::-1]
plt.plot(sorted_attrs, linewidth=2)
plt.xlabel('Feature Rank')
plt.ylabel('Attribution Score')
plt.title('Features Ranked by Attribution')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig(output_dir / f'attribution_distribution_{method}.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: attribution_distribution_{method}.png")
plt.close()

# =========================
# 图 2: Top Features
# =========================
print("Generating top features plot...")

top_k = 30
top_features = feature_ranking[:top_k]
feature_indices = [f[0] for f in top_features]
feature_scores = [f[1] for f in top_features]

# Check which are spatial
spatial_ids = set(spatial_comparison.get('spatial_feature_ids', []))
colors = ['red' if idx in spatial_ids else 'blue' for idx in feature_indices]

plt.figure(figsize=(14, 6))
bars = plt.bar(range(top_k), feature_scores, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Feature Rank')
plt.ylabel('Attribution Score')
plt.title(f'Top {top_k} Features by Attribution ({method})')
plt.xticks(range(top_k), [f"F{idx}" for idx in feature_indices], rotation=90)
plt.grid(True, alpha=0.3, axis='y')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Spatial Features'),
    Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='Non-Spatial Features')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(output_dir / f'top_features_{method}.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: top_features_{method}.png")
plt.close()

# =========================
# 图 3: Spatial vs Non-Spatial Comparison
# =========================
if spatial_comparison:
    print("Generating spatial comparison plot...")
    
    spatial_ids = spatial_comparison['spatial_feature_ids']
    non_spatial_ids = [i for i in range(len(avg_attributions)) if i not in spatial_ids]
    
    spatial_scores = [avg_attributions[i] for i in spatial_ids]
    non_spatial_scores = [avg_attributions[i] for i in non_spatial_ids]
    
    plt.figure(figsize=(10, 6))
    
    # Box plot
    data_to_plot = [spatial_scores, non_spatial_scores]
    labels = [f'Spatial\n(n={len(spatial_scores)})', 
              f'Non-Spatial\n(n={len(non_spatial_scores)})']
    
    bp = plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # Color boxes
    colors = ['red', 'blue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    plt.ylabel('Attribution Score')
    plt.title(f'Spatial vs Non-Spatial Features ({method})')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    spatial_mean = spatial_comparison['spatial_attribution_mean']
    non_spatial_mean = spatial_comparison['non_spatial_attribution_mean']
    ratio = spatial_comparison['spatial_vs_nonspatial_ratio']
    
    plt.text(0.5, 0.95, f'Ratio: {ratio:.2f}x', 
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='top', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'spatial_comparison_{method}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: spatial_comparison_{method}.png")
    plt.close()

# =========================
# 图 4: Attribution Heatmap (Top Features)
# =========================
print("Generating attribution heatmap...")

# Load detailed results if available
detailed_file = output_dir / f"detailed_attributions_{method}_{results['timestamp']}.npz"
if detailed_file.exists():
    try:
        detailed_data = np.load(detailed_file)
        per_sample_attrs = detailed_data['per_sample_attributions']
        correct_mask = detailed_data['correct_mask']
        
        # Check data shape
        print(f"  per_sample_attrs shape: {per_sample_attrs.shape}")
        
        # Handle different data formats
        if len(per_sample_attrs.shape) == 1:
            # Data is flattened, skip heatmap
            print("  Warning: Attribution data is 1D, skipping heatmap")
        elif per_sample_attrs.shape[1] < 50:
            # Not enough features
            print(f"  Warning: Only {per_sample_attrs.shape[1]} features, skipping heatmap")
        else:
            # Select top features
            top_50_features = [f[0] for f in feature_ranking[:50]]
            
            # Ensure indices are valid
            max_feature_idx = per_sample_attrs.shape[1] - 1
            top_50_features = [f for f in top_50_features if f <= max_feature_idx]
            
            if len(top_50_features) == 0:
                print("  Warning: No valid feature indices, skipping heatmap")
            else:
                # Get attributions for top features
                top_attrs = per_sample_attrs[:, top_50_features]
                
                # Separate correct and incorrect predictions
                correct_attrs = top_attrs[correct_mask]
                wrong_attrs = top_attrs[~correct_mask]
                
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                
                # Correct predictions
                if len(correct_attrs) > 0:
                    im1 = axes[0].imshow(correct_attrs.T, aspect='auto', cmap='RdBu_r', 
                                        vmin=-np.abs(top_attrs).max(), vmax=np.abs(top_attrs).max())
                    axes[0].set_xlabel('Sample Index')
                    axes[0].set_ylabel('Feature Index (Top 50)')
                    axes[0].set_title(f'Correct Predictions (n={len(correct_attrs)})')
                    plt.colorbar(im1, ax=axes[0])
                
                # Wrong predictions
                if len(wrong_attrs) > 0:
                    im2 = axes[1].imshow(wrong_attrs.T, aspect='auto', cmap='RdBu_r',
                                        vmin=-np.abs(top_attrs).max(), vmax=np.abs(top_attrs).max())
                    axes[1].set_xlabel('Sample Index')
                    axes[1].set_ylabel('Feature Index (Top 50)')
                    axes[1].set_title(f'Wrong Predictions (n={len(wrong_attrs)})')
                    plt.colorbar(im2, ax=axes[1])
                
                plt.tight_layout()
                plt.savefig(output_dir / f'attribution_heatmap_{method}.png', dpi=150, bbox_inches='tight')
                print(f"✓ Saved: attribution_heatmap_{method}.png")
                plt.close()
    except Exception as e:
        print(f"  Warning: Failed to generate heatmap: {e}")
else:
    print("  Warning: Detailed attribution file not found, skipping heatmap")

# =========================
# 图 5: Feature Overlap Analysis
# =========================
if spatial_comparison and 'top_k_overlap' in spatial_comparison:
    print("Generating overlap analysis plot...")
    
    n_spatial = spatial_comparison['n_spatial_features']
    overlap = spatial_comparison['top_k_overlap']
    overlap_pct = spatial_comparison['top_k_overlap_pct']
    
    # Create Venn-like visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data
    categories = [f'Top-{n_spatial}\nby Attribution', 
                  f'Identified\nSpatial Features',
                  'Overlap']
    values = [n_spatial - overlap, n_spatial - overlap, overlap]
    colors = ['skyblue', 'lightcoral', 'purple']
    
    # Bar plot
    x_pos = np.arange(len(categories))
    bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/n_spatial*100:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Features')
    ax.set_title(f'Feature Overlap Analysis ({method})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add overlap percentage as text
    ax.text(0.5, 0.95, f'Overlap: {overlap}/{n_spatial} ({overlap_pct:.1f}%)',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'feature_overlap_{method}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: feature_overlap_{method}.png")
    plt.close()

# =========================
# 总结
# =========================
print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
print(f"\nAll plots saved to: {output_dir}/")
print("\nGenerated plots:")
print(f"  1. attribution_distribution_{method}.png - Feature attribution distribution")
print(f"  2. top_features_{method}.png - Top features by attribution")
if spatial_comparison:
    print(f"  3. spatial_comparison_{method}.png - Spatial vs non-spatial comparison")
if detailed_file.exists():
    print(f"  4. attribution_heatmap_{method}.png - Per-sample attribution heatmap")
if spatial_comparison and 'top_k_overlap' in spatial_comparison:
    print(f"  5. feature_overlap_{method}.png - Feature overlap analysis")

print("\n" + "="*60)


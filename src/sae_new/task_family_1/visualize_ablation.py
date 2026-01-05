"""
可视化 SAE Feature Ablation 结果
生成论文级别的图表
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Visualize ablation results')
parser.add_argument("--results_file", "-r", type=str, required=True,
                    help="Path to ablation_results_*.json file")
parser.add_argument("--output_dir", "-o", type=str, default=None,
                    help="Output directory for figures (default: same as results)")
args = parser.parse_args()

# =========================
# 加载结果
# =========================
results_file = Path(args.results_file)
if not results_file.exists():
    print(f"Error: {results_file} not found!")
    exit(1)

with open(results_file, 'r') as f:
    results = json.load(f)

# 输出目录
if args.output_dir is None:
    output_dir = results_file.parent
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Visualizing Ablation Results")
print("="*60)
print(f"Results file: {results_file}")
print(f"Output dir: {output_dir}")

# 提取数据
config = results['config']
res = results['results']

conditions = ['Baseline', 'Random\nAblation', 'Spatial\nAblation']
accuracies = [
    res['baseline']['accuracy'],
    res['random_ablation']['accuracy'],
    res['spatial_ablation']['accuracy'],
]
colors = ['#2ecc71', '#f39c12', '#e74c3c']

# =========================
# 图 1: Accuracy Bar Chart
# =========================
fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(conditions, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{acc:.3f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim(0, min(1.0, max(accuracies) * 1.15))
ax.set_title(f'SAE Feature Ablation Results (Layer {config["layer"]})',
             fontsize=16, fontweight='bold', pad=20)

# 添加网格
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
output_file = output_dir / f"ablation_accuracy_bar.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# =========================
# 图 2: Accuracy Drop
# =========================
baseline_acc = res['baseline']['accuracy']
spatial_drop = (baseline_acc - res['spatial_ablation']['accuracy']) * 100
random_drop = (baseline_acc - res['random_ablation']['accuracy']) * 100

fig, ax = plt.subplots(figsize=(8, 6))

drops = [random_drop, spatial_drop]
labels = ['Random\nAblation', 'Spatial\nAblation']
colors_drop = ['#f39c12', '#e74c3c']

bars = ax.bar(labels, drops, color=colors_drop, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar, drop in zip(bars, drops):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{drop:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Accuracy Drop (%)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(drops) * 1.2)
ax.set_title(f'Ablation Effect Size (Baseline: {baseline_acc:.3f})',
             fontsize=16, fontweight='bold', pad=20)

# 添加基线
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
output_file = output_dir / f"ablation_drop.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# =========================
# 图 3: 对比图（Side-by-side）
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：Accuracy
ax1 = axes[0]
bars = ax1.bar(conditions, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax1.set_ylim(0, min(1.0, max(accuracies) * 1.15))
ax1.set_title('(a) Accuracy', fontsize=14, fontweight='bold')
ax1.yaxis.grid(True, alpha=0.3)
ax1.set_axisbelow(True)

# 右图：Drop
ax2 = axes[1]
bars = ax2.bar(labels, drops, color=colors_drop, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, drop in zip(bars, drops):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{drop:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy Drop (%)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(drops) * 1.2)
ax2.set_title('(b) Ablation Effect', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.yaxis.grid(True, alpha=0.3)
ax2.set_axisbelow(True)

plt.suptitle(f'SAE Feature Ablation Analysis (Layer {config["layer"]}, '
             f'{config["n_spatial_features"]} spatial features)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
output_file = output_dir / f"ablation_combined.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# =========================
# 图 4: 样本级别分析（如果有详细数据）
# =========================
if 'examples' in results:
    print("\nGenerating sample-level analysis...")
    
    # 统计每种条件下正确/错误的样本数
    baseline_details = results['examples']['baseline']
    spatial_details = results['examples']['spatial_ablation']
    random_details = results['examples']['random_ablation']
    
    # 计算 flip 情况
    n_samples = len(baseline_details)
    
    # Baseline 正确 -> Spatial Ablation 错误
    baseline_correct_spatial_wrong = sum(
        1 for b, s in zip(baseline_details, spatial_details)
        if b['correct'] and not s['correct']
    )
    
    # Baseline 正确 -> Random Ablation 错误
    baseline_correct_random_wrong = sum(
        1 for b, r in zip(baseline_details, random_details)
        if b['correct'] and not r['correct']
    )
    
    # 绘制 flip 统计
    fig, ax = plt.subplots(figsize=(8, 6))
    
    flip_data = [baseline_correct_random_wrong, baseline_correct_spatial_wrong]
    flip_labels = ['Random\nAblation', 'Spatial\nAblation']
    colors_flip = ['#f39c12', '#e74c3c']
    
    bars = ax.bar(flip_labels, flip_data, color=colors_flip, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, flip_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{count}/{n_samples}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('# Samples Flipped from Correct to Wrong', 
                  fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(flip_data) * 1.2)
    ax.set_title('Sample Flip Analysis\n(Baseline Correct → Ablation Wrong)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_file = output_dir / f"ablation_flips.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

# =========================
# 生成 LaTeX 表格代码
# =========================
print("\n" + "="*60)
print("LaTeX Table Code:")
print("="*60)

latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{SAE Feature Ablation Results on Spatial Reasoning Task}}
\\label{{tab:ablation_results}}
\\begin{{tabular}}{{lccc}}
\\toprule
Condition & Accuracy & Accuracy Drop & Relative Drop \\\\
\\midrule
Baseline (No Ablation) & {res['baseline']['accuracy']:.3f} & - & - \\\\
Random Ablation & {res['random_ablation']['accuracy']:.3f} & {random_drop:.1f}\\% & {res['random_ablation']['drop_pct']:.1f}\\% \\\\
Spatial Ablation & {res['spatial_ablation']['accuracy']:.3f} & {spatial_drop:.1f}\\% & {res['spatial_ablation']['drop_pct']:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item Note: Model: {config['model_name']}, Layer {config['layer']}, 
{config['n_spatial_features']} spatial features ablated, 
{config['n_eval_samples']} evaluation samples.
\\end{{tablenotes}}
\\end{{table}}
"""

print(latex_table)

# 保存 LaTeX 代码
latex_file = output_dir / "ablation_table.tex"
with open(latex_file, 'w') as f:
    f.write(latex_table)
print(f"\n✓ LaTeX table saved to: {latex_file}")

# =========================
# 总结
# =========================
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Baseline accuracy: {res['baseline']['accuracy']:.3f}")
print(f"Spatial ablation accuracy: {res['spatial_ablation']['accuracy']:.3f} ({spatial_drop:.1f}% drop)")
print(f"Random ablation accuracy: {res['random_ablation']['accuracy']:.3f} ({random_drop:.1f}% drop)")
print(f"\nEffect size (spatial - random): {spatial_drop - random_drop:.1f}%")

if spatial_drop > random_drop * 2:
    print("\n✓ Strong evidence: Spatial features are causally important!")
elif spatial_drop > random_drop * 1.5:
    print("\n~ Moderate evidence: Spatial features may be important.")
else:
    print("\n✗ Weak evidence: No clear causal role for spatial features.")

print("\n" + "="*60)
print("All figures saved to:", output_dir)
print("="*60)




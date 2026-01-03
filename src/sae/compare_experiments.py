"""
比较多个 SAE 实验的结果
用于可视化网格搜索或不同配置的对比
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description='Compare multiple SAE experiments')
parser.add_argument('--exp_dirs', '-e', nargs='+', required=True,
                    help='List of experiment directories')
parser.add_argument('--output', '-o', type=str, default='comparison.png',
                    help='Output figure path')
args = parser.parse_args()

# 收集所有实验的数据
experiments = []

for exp_dir in args.exp_dirs:
    exp_path = Path(exp_dir)
    
    # 读取配置
    config_file = exp_path / 'config.json'
    history_file = exp_path / 'training_history.json'
    probe_file = exp_path / 'analysis' / 'probe_results.json'
    
    if not all([config_file.exists(), history_file.exists()]):
        print(f"Warning: Skipping {exp_dir} (missing files)")
        continue
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    probe_r2 = None
    if probe_file.exists():
        with open(probe_file, 'r') as f:
            probe = json.load(f)
            probe_r2 = probe.get('r2_full', None)
    
    exp_name = f"F{config['n_features']}_L1{config['l1_coefficient']:.0e}"
    
    experiments.append({
        'name': exp_name,
        'dir': exp_dir,
        'config': config,
        'history': history,
        'probe_r2': probe_r2,
    })

if len(experiments) == 0:
    print("Error: No valid experiments found")
    sys.exit(1)

print(f"Comparing {len(experiments)} experiments")

# 创建对比图
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Training Loss
ax = axes[0, 0]
for exp in experiments:
    ax.plot(exp['history']['train_loss'], label=exp['name'], marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Train Loss')
ax.set_title('Training Loss')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Test Loss
ax = axes[0, 1]
for exp in experiments:
    ax.plot(exp['history']['test_loss'], label=exp['name'], marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Loss')
ax.set_title('Test Loss')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. L0 (Sparsity)
ax = axes[0, 2]
for exp in experiments:
    ax.plot(exp['history']['test_l0'], label=exp['name'], marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('L0 (# active features)')
ax.set_title('Sparsity (L0)')
ax.axhline(y=20, color='r', linestyle='--', alpha=0.3, label='Target min')
ax.axhline(y=100, color='r', linestyle='--', alpha=0.3, label='Target max')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Cosine Similarity
ax = axes[1, 0]
for exp in experiments:
    ax.plot(exp['history']['test_cos_sim'], label=exp['name'], marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Reconstruction Similarity')
ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.3, label='Target')
ax.set_ylim([0.8, 1.0])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 5. Final Metrics Comparison (Bar chart)
ax = axes[1, 1]
exp_names = [exp['name'] for exp in experiments]
final_l0 = [exp['history']['test_l0'][-1] for exp in experiments]
final_cos = [exp['history']['test_cos_sim'][-1] * 100 for exp in experiments]  # Scale to 0-100

x = np.arange(len(exp_names))
width = 0.35

ax.bar(x - width/2, final_l0, width, label='L0', alpha=0.7)
ax.bar(x + width/2, final_cos, width, label='CosSim×100', alpha=0.7)

ax.set_xlabel('Experiment')
ax.set_ylabel('Value')
ax.set_title('Final Metrics (Last Epoch)')
ax.set_xticks(x)
ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 6. Probe R² Comparison (if available)
ax = axes[1, 2]
probe_r2_values = [exp['probe_r2'] for exp in experiments if exp['probe_r2'] is not None]

if len(probe_r2_values) > 0:
    valid_exps = [exp for exp in experiments if exp['probe_r2'] is not None]
    exp_names = [exp['name'] for exp in valid_exps]
    
    bars = ax.bar(exp_names, probe_r2_values, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Probe R²')
    ax.set_title('Feature Probe Performance')
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0.27, color='r', linestyle='--', alpha=0.5, label='MLP-out probe (baseline)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 标注最佳
    best_idx = np.argmax(probe_r2_values)
    bars[best_idx].set_color('green')
    bars[best_idx].set_edgecolor('darkgreen')
    bars[best_idx].set_linewidth(2)
else:
    ax.text(0.5, 0.5, 'No probe results available\nRun analyze_features.py first',
            ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')

plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"\nComparison plot saved to: {args.output}")

# 打印总结表格
print("\n" + "="*80)
print("EXPERIMENT COMPARISON SUMMARY")
print("="*80)
print(f"{'Experiment':<20} {'L0':<8} {'CosSim':<10} {'Probe R²':<10} {'Status'}")
print("-"*80)

for exp in experiments:
    l0 = exp['history']['test_l0'][-1]
    cos = exp['history']['test_cos_sim'][-1]
    r2 = exp['probe_r2'] if exp['probe_r2'] is not None else float('nan')
    
    # 状态评估
    status = []
    if 20 <= l0 <= 100:
        status.append("✓ Sparse")
    else:
        status.append("✗ L0" + (" high" if l0 > 100 else " low"))
    
    if cos >= 0.95:
        status.append("✓ Recon")
    else:
        status.append("✗ Recon")
    
    if not np.isnan(r2) and r2 >= 0.20:
        status.append("✓ Probe")
    elif not np.isnan(r2):
        status.append("✗ Probe")
    
    status_str = " ".join(status)
    
    print(f"{exp['name']:<20} {l0:<8.1f} {cos:<10.4f} {r2:<10.4f} {status_str}")

print("="*80)
print("\nLegend:")
print("  ✓ Sparse: L0 in [20, 100]")
print("  ✓ Recon: Cosine Similarity >= 0.95")
print("  ✓ Probe: R² >= 0.20")
print("\nRecommendation: Choose experiment with all ✓ marks and highest Probe R²")
print("="*80)



#!/usr/bin/env python3
"""
分析所有层的SAE训练结果
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_layer_results(layer_dir):
    """加载单个层的结果"""
    results = {}
    
    # 加载配置
    with open(layer_dir / "config.json", 'r') as f:
        results['config'] = json.load(f)
    
    # 加载训练历史
    with open(layer_dir / "training_history.json", 'r') as f:
        results['training_history'] = json.load(f)
    
    # 加载探针结果
    probe_file = layer_dir / "analysis" / "probe_results.json"
    if probe_file.exists():
        with open(probe_file, 'r') as f:
            results['probe_results'] = json.load(f)
    
    # 加载维度特征
    dim_file = layer_dir / "analysis" / "dimension_features.json"
    if dim_file.exists():
        with open(dim_file, 'r') as f:
            results['dimension_features'] = json.load(f)
    
    return results

def analyze_all_layers(base_dir):
    """分析所有层的结果"""
    base_path = Path(base_dir)
    
    # 查找所有层目录
    layer_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('L')])
    
    all_results = {}
    for layer_dir in layer_dirs:
        # 从目录名提取层号
        layer_name = layer_dir.name.split('_')[0]
        layer_num = int(layer_name[1:])
        
        print(f"Loading {layer_name}...")
        try:
            all_results[layer_num] = load_layer_results(layer_dir)
        except Exception as e:
            print(f"  Error loading {layer_name}: {e}")
    
    return all_results

def create_summary_dataframe(all_results):
    """创建汇总数据框"""
    summary_data = []
    
    for layer, results in sorted(all_results.items()):
        row = {'layer': layer}
        
        # 训练历史的最终值
        th = results['training_history']
        row['final_train_loss'] = th['train_loss'][-1]
        row['final_train_mse'] = th['train_mse'][-1]
        row['final_train_l1'] = th['train_l1'][-1]
        row['final_train_l0'] = th['train_l0'][-1]
        row['final_test_loss'] = th['test_loss'][-1]
        row['final_test_mse'] = th['test_mse'][-1]
        row['final_test_l1'] = th['test_l1'][-1]
        row['final_test_l0'] = th['test_l0'][-1]
        
        # 探针性能
        if 'probe_results' in results:
            pr = results['probe_results']
            row['probe_r2_full'] = pr.get('r2_full', None)
            if 'r2_top_k' in pr:
                row['probe_r2_top10'] = pr['r2_top_k'].get('10', None)
                row['probe_r2_top20'] = pr['r2_top_k'].get('20', None)
                row['probe_r2_top50'] = pr['r2_top_k'].get('50', None)
        
        # 维度特征统计
        if 'dimension_features' in results:
            df = results['dimension_features']
            row['n_X_positive'] = len(df.get('X_positive', []))
            row['n_X_negative'] = len(df.get('X_negative', []))
            row['n_Y_positive'] = len(df.get('Y_positive', []))
            row['n_Y_negative'] = len(df.get('Y_negative', []))
            
            # 计算最强相关性
            all_corrs = []
            for dim in ['X_positive', 'X_negative', 'Y_positive', 'Y_negative']:
                features = df.get(dim, [])
                if features:
                    all_corrs.extend([abs(f['correlation']) for f in features])
            
            if all_corrs:
                row['max_abs_correlation'] = max(all_corrs)
                row['mean_abs_correlation'] = np.mean(all_corrs)
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def plot_training_metrics(all_results, output_dir):
    """绘制训练指标"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 准备数据
    layers = sorted(all_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics Across Layers', fontsize=16, fontweight='bold')
    
    metrics = [
        ('test_loss', 'Test Loss', axes[0, 0]),
        ('test_mse', 'Test MSE', axes[0, 1]),
        ('test_l1', 'Test L1', axes[1, 0]),
        ('test_l0', 'Test L0 (Sparsity)', axes[1, 1])
    ]
    
    for metric_name, title, ax in metrics:
        values = []
        for layer in layers:
            th = all_results[layer]['training_history']
            values.append(th[metric_name][-1])
        
        ax.plot(layers, values, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics_across_layers.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_metrics_across_layers.png'}")
    plt.close()

def plot_probe_performance(all_results, output_dir):
    """绘制探针性能"""
    output_dir = Path(output_dir)
    
    layers = sorted(all_results.keys())
    r2_full = []
    r2_top10 = []
    r2_top20 = []
    r2_top50 = []
    
    for layer in layers:
        if 'probe_results' in all_results[layer]:
            pr = all_results[layer]['probe_results']
            r2_full.append(pr.get('r2_full', np.nan))
            if 'r2_top_k' in pr:
                r2_top10.append(pr['r2_top_k'].get('10', np.nan))
                r2_top20.append(pr['r2_top_k'].get('20', np.nan))
                r2_top50.append(pr['r2_top_k'].get('50', np.nan))
        else:
            r2_full.append(np.nan)
            r2_top10.append(np.nan)
            r2_top20.append(np.nan)
            r2_top50.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(layers, r2_full, marker='o', label='Full Features', linewidth=2, markersize=6)
    ax.plot(layers, r2_top10, marker='s', label='Top 10', linewidth=2, markersize=6)
    ax.plot(layers, r2_top20, marker='^', label='Top 20', linewidth=2, markersize=6)
    ax.plot(layers, r2_top50, marker='d', label='Top 50', linewidth=2, markersize=6)
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Line')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Probe Performance (R²) Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'probe_performance_across_layers.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'probe_performance_across_layers.png'}")
    plt.close()

def plot_correlation_analysis(all_results, output_dir):
    """绘制相关性分析"""
    output_dir = Path(output_dir)
    
    layers = sorted(all_results.keys())
    
    data = {
        'X_positive': [],
        'X_negative': [],
        'Y_positive': [],
        'Y_negative': []
    }
    
    max_corrs = []
    mean_corrs = []
    
    for layer in layers:
        if 'dimension_features' in all_results[layer]:
            df = all_results[layer]['dimension_features']
            
            all_corrs = []
            for dim in data.keys():
                features = df.get(dim, [])
                data[dim].append(len(features))
                if features:
                    all_corrs.extend([abs(f['correlation']) for f in features])
            
            if all_corrs:
                max_corrs.append(max(all_corrs))
                mean_corrs.append(np.mean(all_corrs))
            else:
                max_corrs.append(0)
                mean_corrs.append(0)
    
    # 特征数量堆叠图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：特征数量
    ax1 = axes[0]
    bottom = np.zeros(len(layers))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for (dim, counts), color in zip(data.items(), colors):
        ax1.bar(layers, counts, bottom=bottom, label=dim, color=color, alpha=0.8)
        bottom += counts
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Number of Features', fontsize=12)
    ax1.set_title('Feature Count by Dimension Across Layers', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：相关性强度
    ax2 = axes[1]
    ax2.plot(layers, max_corrs, marker='o', label='Max |Correlation|', linewidth=2, markersize=6)
    ax2.plot(layers, mean_corrs, marker='s', label='Mean |Correlation|', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Absolute Correlation', fontsize=12)
    ax2.set_title('Correlation Strength Across Layers', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_analysis_across_layers.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'correlation_analysis_across_layers.png'}")
    plt.close()

def plot_training_convergence(all_results, output_dir, selected_layers=[0, 7, 14, 21, 27]):
    """绘制选定层的训练收敛曲线"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Convergence for Selected Layers', fontsize=16, fontweight='bold')
    
    metrics = [
        ('test_loss', 'Test Loss', axes[0, 0]),
        ('test_mse', 'Test MSE', axes[0, 1]),
        ('test_l1', 'Test L1', axes[1, 0]),
        ('test_l0', 'Test L0', axes[1, 1])
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_layers)))
    
    for metric_name, title, ax in metrics:
        for layer, color in zip(selected_layers, colors):
            if layer in all_results:
                th = all_results[layer]['training_history']
                values = th[metric_name]
                epochs = range(len(values))
                ax.plot(epochs, values, marker='o', label=f'Layer {layer}', 
                       color=color, linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_convergence_selected_layers.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_convergence_selected_layers.png'}")
    plt.close()

def generate_summary_report(df, all_results, output_dir):
    """生成文本摘要报告"""
    output_dir = Path(output_dir)
    
    report = []
    report.append("=" * 80)
    report.append("SAE Training Results Summary - All Layers")
    report.append("=" * 80)
    report.append("")
    
    # 基本信息
    report.append("## Basic Information")
    report.append(f"Total Layers: {len(all_results)}")
    report.append(f"Model: {all_results[0]['config']['model_name']}")
    report.append(f"Number of Features per Layer: {all_results[0]['config']['n_features']}")
    report.append(f"L1 Coefficient: {all_results[0]['config']['l1_coefficient']}")
    report.append(f"Training Samples: {all_results[0]['config']['train_samples']}")
    report.append(f"Test Samples: {all_results[0]['config']['test_samples']}")
    report.append("")
    
    # 训练指标统计
    report.append("## Training Metrics Statistics")
    report.append("")
    metrics = ['final_test_loss', 'final_test_mse', 'final_test_l1', 'final_test_l0']
    for metric in metrics:
        if metric in df.columns:
            report.append(f"{metric}:")
            report.append(f"  Min: {df[metric].min():.6f} (Layer {df.loc[df[metric].idxmin(), 'layer']:.0f})")
            report.append(f"  Max: {df[metric].max():.6f} (Layer {df.loc[df[metric].idxmax(), 'layer']:.0f})")
            report.append(f"  Mean: {df[metric].mean():.6f}")
            report.append(f"  Std: {df[metric].std():.6f}")
            report.append("")
    
    # 探针性能统计
    report.append("## Probe Performance Statistics")
    report.append("")
    if 'probe_r2_full' in df.columns:
        report.append(f"R² (Full Features):")
        report.append(f"  Min: {df['probe_r2_full'].min():.6f} (Layer {df.loc[df['probe_r2_full'].idxmin(), 'layer']:.0f})")
        report.append(f"  Max: {df['probe_r2_full'].max():.6f} (Layer {df.loc[df['probe_r2_full'].idxmax(), 'layer']:.0f})")
        report.append(f"  Mean: {df['probe_r2_full'].mean():.6f}")
        report.append("")
        
        # 负R²的层
        negative_r2 = df[df['probe_r2_full'] < 0]
        report.append(f"Layers with Negative R²: {len(negative_r2)} out of {len(df)}")
        if len(negative_r2) > 0:
            report.append(f"  Negative R² layers: {negative_r2['layer'].tolist()}")
        report.append("")
    
    # 相关性统计
    if 'max_abs_correlation' in df.columns:
        report.append("## Correlation Statistics")
        report.append("")
        report.append(f"Max Absolute Correlation:")
        report.append(f"  Min: {df['max_abs_correlation'].min():.6f} (Layer {df.loc[df['max_abs_correlation'].idxmin(), 'layer']:.0f})")
        report.append(f"  Max: {df['max_abs_correlation'].max():.6f} (Layer {df.loc[df['max_abs_correlation'].idxmax(), 'layer']:.0f})")
        report.append(f"  Mean: {df['max_abs_correlation'].mean():.6f}")
        report.append("")
    
    # 最佳和最差层
    report.append("## Best and Worst Performing Layers")
    report.append("")
    if 'probe_r2_full' in df.columns:
        best_layers = df.nlargest(5, 'probe_r2_full')
        worst_layers = df.nsmallest(5, 'probe_r2_full')
        
        report.append("Top 5 Layers by R²:")
        for _, row in best_layers.iterrows():
            report.append(f"  Layer {row['layer']:.0f}: R² = {row['probe_r2_full']:.6f}")
        report.append("")
        
        report.append("Bottom 5 Layers by R²:")
        for _, row in worst_layers.iterrows():
            report.append(f"  Layer {row['layer']:.0f}: R² = {row['probe_r2_full']:.6f}")
        report.append("")
    
    # 特征维度分布
    if 'n_X_positive' in df.columns:
        report.append("## Feature Dimension Distribution")
        report.append("")
        report.append(f"Total X_positive features: {df['n_X_positive'].sum():.0f}")
        report.append(f"Total X_negative features: {df['n_X_negative'].sum():.0f}")
        report.append(f"Total Y_positive features: {df['n_Y_positive'].sum():.0f}")
        report.append(f"Total Y_negative features: {df['n_Y_negative'].sum():.0f}")
        report.append("")
    
    report.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report)
    with open(output_dir / 'summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Saved: {output_dir / 'summary_report.txt'}")
    return report_text

def main():
    base_dir = "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae_saelens/sae_results_all_layers_task_1"
    output_dir = Path(base_dir) / "analysis_summary"
    output_dir.mkdir(exist_ok=True)
    
    print("Loading all layer results...")
    all_results = analyze_all_layers(base_dir)
    
    print(f"\nLoaded {len(all_results)} layers")
    print("\nCreating summary dataframe...")
    df = create_summary_dataframe(all_results)
    
    # 保存汇总表格
    df.to_csv(output_dir / 'summary_table.csv', index=False)
    print(f"Saved: {output_dir / 'summary_table.csv'}")
    
    # 生成可视化
    print("\nGenerating visualizations...")
    plot_training_metrics(all_results, output_dir)
    plot_probe_performance(all_results, output_dir)
    plot_correlation_analysis(all_results, output_dir)
    plot_training_convergence(all_results, output_dir)
    
    # 生成文本报告
    print("\nGenerating summary report...")
    report = generate_summary_report(df, all_results, output_dir)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - summary_table.csv")
    print("  - summary_report.txt")
    print("  - training_metrics_across_layers.png")
    print("  - probe_performance_across_layers.png")
    print("  - correlation_analysis_across_layers.png")
    print("  - training_convergence_selected_layers.png")
    print("\n" + "="*80)
    
    # 打印摘要报告
    print("\n\n" + report)

if __name__ == "__main__":
    main()


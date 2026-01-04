#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析方向推理测试结果
"""

import json
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List


def analyze_results(results_file: str):
    """
    分析测试结果
    
    Args:
        results_file: 结果文件路径
    """
    # 读取结果
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("="*60)
    print("Orientation Reasoning Test Results Analysis")
    print("="*60)
    print(f"\nModel: {data['model']}")
    print(f"Total samples: {data['total_samples']}")
    print(f"Correct predictions: {data['correct_count']}")
    print(f"Overall accuracy: {data['overall_accuracy']:.2%}")
    
    # 按步骤数分析
    print("\n" + "="*60)
    print("Accuracy by Number of Steps:")
    print("="*60)
    
    step_accuracy = data['step_accuracy']
    for num_steps in sorted([int(k) for k in step_accuracy.keys()]):
        stats = step_accuracy[str(num_steps)]
        print(f"  {num_steps:2d} steps: {stats['accuracy']:6.2%} "
              f"({stats['correct']:3d}/{stats['total']:3d})")
    
    # 分析错误案例
    print("\n" + "="*60)
    print("Error Analysis:")
    print("="*60)
    
    errors = [r for r in data['results'] if not r['is_correct']]
    print(f"\nTotal errors: {len(errors)}")
    
    if errors:
        # 按步骤数统计错误
        error_by_steps = defaultdict(int)
        for error in errors:
            error_by_steps[error['num_steps']] += 1
        
        print("\nErrors by number of steps:")
        for num_steps in sorted(error_by_steps.keys()):
            print(f"  {num_steps} steps: {error_by_steps[num_steps]} errors")
        
        # 显示一些错误示例
        print("\n" + "-"*60)
        print("Sample Errors:")
        print("-"*60)
        for i, error in enumerate(errors[:5]):  # 显示前5个错误
            print(f"\nError {i+1}:")
            print(f"Sample ID: {error['id']}")
            print(f"Steps: {error['num_steps']}")
            print(f"Question:\n{error['question']}")
            print(f"Correct answer: {error['correct_option']}")
            print(f"Predicted answer: {error['predicted_answer']}")
            print(f"Response:\n{error['response_text'][:200]}...")
            print("-"*60)
    
    # 绘制图表
    plot_accuracy_by_steps(step_accuracy, data['model'])


def plot_accuracy_by_steps(step_accuracy: Dict, model_name: str):
    """
    绘制按步骤数的准确率图表
    
    Args:
        step_accuracy: 步骤准确率字典
        model_name: 模型名称
    """
    steps = sorted([int(k) for k in step_accuracy.keys()])
    accuracies = [step_accuracy[str(s)]['accuracy'] for s in steps]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Steps', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Orientation Reasoning Accuracy by Number of Steps\nModel: {model_name}', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # 添加数值标签
    for i, (step, acc) in enumerate(zip(steps, accuracies)):
        plt.text(step, acc + 0.02, f'{acc:.2%}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('orientation_accuracy_by_steps.png', dpi=300)
    print("\n" + "="*60)
    print("Plot saved to: orientation_accuracy_by_steps.png")
    print("="*60)


def compare_with_spatial_reasoning(orientation_file: str, spatial_file: str):
    """
    比较方向推理和空间推理的结果
    
    Args:
        orientation_file: 方向推理结果文件
        spatial_file: 空间推理结果文件
    """
    # 读取两个结果文件
    with open(orientation_file, 'r', encoding='utf-8') as f:
        orientation_data = json.load(f)
    
    with open(spatial_file, 'r', encoding='utf-8') as f:
        spatial_data = json.load(f)
    
    print("\n" + "="*60)
    print("Comparison: Orientation vs Spatial Reasoning")
    print("="*60)
    
    print(f"\nOrientation Reasoning:")
    print(f"  Model: {orientation_data['model']}")
    print(f"  Overall accuracy: {orientation_data['overall_accuracy']:.2%}")
    
    print(f"\nSpatial Reasoning:")
    print(f"  Model: {spatial_data['model']}")
    print(f"  Overall accuracy: {spatial_data['overall_accuracy']:.2%}")
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    
    categories = ['Orientation\nReasoning', 'Spatial\nReasoning']
    accuracies = [
        orientation_data['overall_accuracy'],
        spatial_data['overall_accuracy']
    ]
    
    bars = plt.bar(categories, accuracies, color=['#3498db', '#e74c3c'], alpha=0.7)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Comparison of Reasoning Task Performance', fontsize=14)
    plt.ylim([0, 1.05])
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('orientation_vs_spatial_comparison.png', dpi=300)
    print("\nComparison plot saved to: orientation_vs_spatial_comparison.png")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_orientation_results.py <results_file> [spatial_results_file]")
        print("\nExample:")
        print("  python analyze_orientation_results.py orientation_test_results.json")
        print("  python analyze_orientation_results.py orientation_test_results.json api_test_results.json")
        return
    
    results_file = sys.argv[1]
    
    # 分析方向推理结果
    analyze_results(results_file)
    
    # 如果提供了空间推理结果文件，进行对比
    if len(sys.argv) >= 3:
        spatial_file = sys.argv[2]
        try:
            compare_with_spatial_reasoning(results_file, spatial_file)
        except FileNotFoundError:
            print(f"\nWarning: Spatial reasoning file '{spatial_file}' not found. Skipping comparison.")
        except Exception as e:
            print(f"\nError during comparison: {str(e)}")


if __name__ == "__main__":
    main()


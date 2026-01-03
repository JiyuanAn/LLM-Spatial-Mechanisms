#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间过程执行数据集使用示例
演示如何加载和使用生成的数据集
"""

import json
from generate_spatial_procedure_EN import SpatialProcedureGenerator


def load_dataset(filename):
    """加载数据集"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def show_sample_details(sample):
    """显示样本详情"""
    print("="*60)
    print(f"Sample ID: {sample['id']}")
    print(f"Number of steps: {sample['num_steps']}")
    print(f"\nQuestion:\n{sample['question']}")
    
    print(f"\nOptions:")
    for i, option in enumerate(sample['options']):
        label = chr(65 + i)
        marker = " ✓" if label == sample['correct_option'] else ""
        print(f"  {label}. {option}{marker}")
    
    print(f"\nCorrect Answer: {sample['correct_option']} ({sample['answer']})")
    
    print(f"\nAction Sequence:")
    for i, action in enumerate(sample['actions'], 1):
        print(f"  {i}. {action['description']}")
        print(f"     Type: {action['type']}, Params: {action['params']}")
    
    print(f"\nStart Position: {sample['start_position']}")
    print(f"Final Position: {sample['final_position']}")
    print(f"Target Vector: {sample['target']}")
    print("="*60)


def verify_solution(sample):
    """验证样本的解答是否正确"""
    generator = SpatialProcedureGenerator()
    
    # 从起始位置开始
    position = tuple(sample['start_position'])
    
    # 应用每个操作
    print("\nVerifying step by step:")
    print(f"Start: {position}")
    
    for i, action_data in enumerate(sample['actions'], 1):
        from generate_spatial_procedure_EN import SpatialAction
        action = SpatialAction(
            action_type=action_data['type'],
            params=action_data['params']
        )
        position = generator.apply_action(position, action)
        print(f"Step {i} ({action_data['type']}): {position}")
    
    # 四舍五入
    final_position = tuple(round(coord, 2) for coord in position)
    expected_position = tuple(sample['final_position'])
    
    print(f"\nFinal Position: {final_position}")
    print(f"Expected: {expected_position}")
    
    is_correct = final_position == expected_position
    print(f"Verification: {'✓ PASS' if is_correct else '✗ FAIL'}")
    
    return is_correct


def main():
    """主函数"""
    print("="*60)
    print("Spatial Procedure Execution Dataset - Usage Example")
    print("="*60)
    
    # 加载测试数据集
    print("\nLoading test dataset...")
    dataset = load_dataset('spatial_procedure_dataset_EN_test.json')
    print(f"Loaded {len(dataset)} samples")
    
    # 显示第一个样本
    print("\n" + "="*60)
    print("Example 1: Simple Sample")
    print("="*60)
    sample = dataset[0]
    show_sample_details(sample)
    
    # 验证解答
    verify_solution(sample)
    
    # 显示一个复杂样本
    print("\n" + "="*60)
    print("Example 2: Complex Sample")
    print("="*60)
    # 找一个步骤较多的样本
    complex_samples = [s for s in dataset if s['num_steps'] >= 8]
    if complex_samples:
        sample = complex_samples[0]
        show_sample_details(sample)
        verify_solution(sample)
    
    # 显示带提示词的样本
    print("\n" + "="*60)
    print("Example 3: Sample with Prompt")
    print("="*60)
    try:
        dataset_with_prompt = load_dataset('spatial_procedure_dataset_EN_test_with_prompt.json')
        sample = dataset_with_prompt[0]
        print(f"Sample ID: {sample['id']}")
        print("\nPrompt:")
        print("-"*60)
        print(sample['prompt'])
        print("-"*60)
        print(f"\nCorrect Answer: {sample['correct_option']} ({sample['answer']})")
    except FileNotFoundError:
        print("Dataset with prompts not found. Run add_prompt_spatial_procedure.py first.")
    
    # 统计信息
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    from collections import Counter
    
    step_counts = Counter(s['num_steps'] for s in dataset)
    print(f"\nSamples by number of steps:")
    for steps in sorted(step_counts.keys()):
        print(f"  {steps} steps: {step_counts[steps]} samples")
    
    # 统计操作类型
    action_types = []
    for sample in dataset:
        action_types.extend(a['type'] for a in sample['actions'])
    
    action_counts = Counter(action_types)
    print(f"\nAction types distribution:")
    for action_type in sorted(action_counts.keys()):
        print(f"  {action_type}: {action_counts[action_type]} times")


if __name__ == "__main__":
    main()


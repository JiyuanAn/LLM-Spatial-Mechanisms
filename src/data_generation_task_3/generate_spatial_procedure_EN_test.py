#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间过程执行数据集生成器 - 测试集版本
生成较小的测试数据集用于快速测试
"""

import json
import random
from collections import defaultdict
from generate_spatial_procedure_EN import SpatialProcedureGenerator


def main():
    """主函数"""
    # 创建生成器
    generator = SpatialProcedureGenerator(seed=123)  # 使用不同的种子
    
    # 生成测试数据集（100个样本）
    print("Generating spatial procedure execution test dataset...")
    dataset = generator.generate_dataset(
        num_samples=100,
        min_steps=2,
        max_steps=10,
        output_file='spatial_procedure_dataset_EN_test.json'
    )
    
    # 打印一些示例
    print("\n" + "="*60)
    print("Test Dataset Sample Examples:")
    print("="*60)
    
    # 打印不同步骤数的示例
    for num_steps in [2, 5, 8]:
        samples_with_n_steps = [s for s in dataset if s['num_steps'] == num_steps]
        if samples_with_n_steps:
            generator.print_sample(samples_with_n_steps[0])
    
    # 统计信息
    print("\n" + "="*60)
    print("Test Dataset Statistics:")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    
    step_distribution = defaultdict(int)
    for sample in dataset:
        step_distribution[sample['num_steps']] += 1
    
    print("\nDistribution by number of steps:")
    for steps in sorted(step_distribution.keys()):
        print(f"  {steps} steps: {step_distribution[steps]} samples")
    
    # 统计操作类型分布
    action_type_count = defaultdict(int)
    for sample in dataset:
        for action in sample['actions']:
            action_type_count[action['type']] += 1
    
    print("\nDistribution by action type:")
    for action_type in sorted(action_type_count.keys()):
        print(f"  {action_type}: {action_type_count[action_type]} actions")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为空间过程执行数据集添加提示词
"""

import json


def add_prompt_to_dataset(input_file: str, output_file: str):
    """
    为数据集添加提示词
    
    Args:
        input_file: 输入的数据集文件
        output_file: 输出的带提示词的数据集文件
    """
    # 读取原始数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 定义提示词模板
    prompt_template = """You are an AI assistant that helps solve spatial procedure execution problems.

Please carefully read the following problem and choose the correct answer from the options provided.

Problem:
{question}

Options:
{options_text}

Please provide your answer in the following format:
Answer: [Your choice (A/B/C/D)]
Explanation: [Brief explanation of your reasoning]"""
    
    # 为每个样本添加提示词
    for sample in dataset:
        # 格式化选项文本
        options_text = "\n".join([
            f"{chr(65 + i)}. {option}" 
            for i, option in enumerate(sample['options'])
        ])
        
        # 生成带提示词的问题
        prompt = prompt_template.format(
            question=sample['question'],
            options_text=options_text
        )
        
        sample['prompt'] = prompt
    
    # 保存带提示词的数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Added prompts to {len(dataset)} samples")
    print(f"Dataset with prompts saved to {output_file}")


def main():
    """主函数"""
    import os
    
    # 检查并处理完整数据集
    if os.path.exists('spatial_procedure_dataset_EN.json'):
        print("Adding prompts to full dataset...")
        add_prompt_to_dataset(
            input_file='spatial_procedure_dataset_EN.json',
            output_file='spatial_procedure_dataset_EN_with_prompt.json'
        )
    else:
        print("Full dataset not found, skipping...")
    
    # 检查并处理测试数据集
    if os.path.exists('spatial_procedure_dataset_EN_test.json'):
        print("\nAdding prompts to test dataset...")
        add_prompt_to_dataset(
            input_file='spatial_procedure_dataset_EN_test.json',
            output_file='spatial_procedure_dataset_EN_test_with_prompt.json'
        )
    else:
        print("Test dataset not found, skipping...")
    
    # 打印一个示例（优先使用完整数据集）
    sample_file = None
    if os.path.exists('spatial_procedure_dataset_EN_with_prompt.json'):
        sample_file = 'spatial_procedure_dataset_EN_with_prompt.json'
    elif os.path.exists('spatial_procedure_dataset_EN_test_with_prompt.json'):
        sample_file = 'spatial_procedure_dataset_EN_test_with_prompt.json'
    
    if sample_file:
        with open(sample_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print("\n" + "="*60)
        print("Sample with Prompt:")
        print("="*60)
        print(dataset[0]['prompt'])
        print("\n" + "="*60)
        print(f"Correct Answer: {dataset[0]['correct_option']} ({dataset[0]['answer']})")
        print("="*60)


if __name__ == "__main__":
    main()


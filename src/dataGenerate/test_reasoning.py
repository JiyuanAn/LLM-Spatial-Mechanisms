#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试空间推理数据生成器，验证生成的问题确实需要多跳推理
"""

from generate_spatial_reasoning import SpatialReasoningGenerator

def test_single_sample():
    """测试单个样本生成"""
    generator = SpatialReasoningGenerator(seed=42)
    
    # 测试不同步数的样本
    for num_steps in [3, 5, 7, 10]:
        print(f"\n{'='*70}")
        print(f"Testing {num_steps}-step reasoning:")
        print('='*70)
        
        relations, positions, start_entity, end_entity = generator.generate_chain(num_steps)
        
        # 验证是否需要推理
        needs_reasoning = generator.verify_needs_reasoning(relations, start_entity, end_entity)
        print(f"\nNeeds multi-hop reasoning: {needs_reasoning}")
        
        # 打印关系
        print(f"\nSpatial relations:")
        for i, rel in enumerate(relations, 1):
            print(f"  {i}. {rel}")
        
        # 生成问题和答案
        question, answer = generator.generate_question_answer(relations, positions, start_entity, end_entity)
        
        print(f"\nQuestion:")
        question_lines = question.split('\n')
        for line in question_lines:
            print(f"  {line}")
        
        print(f"\nAnswer: {answer}")
        
        # 验证起点和终点之间没有直接关系
        print(f"\nVerification:")
        print(f"  Query: {start_entity} and {end_entity}")
        has_direct = False
        for rel in relations:
            if (rel.subject == start_entity and rel.object == end_entity) or \
               (rel.subject == end_entity and rel.object == start_entity):
                print(f"  WARNING: Direct relation found: {rel}")
                has_direct = True
        
        if not has_direct:
            print(f"  ✓ No direct   relation between {start_entity} and {end_entity}")
            print(f"  ✓ This question requires {num_steps}-step reasoning!")

def test_batch_generation():
    """测试批量生成并统计"""
    print(f"\n\n{'='*70}")
    print("Testing batch generation (10 samples)...")
    print('='*70)
    
    generator = SpatialReasoningGenerator(seed=123)
    dataset = generator.generate_dataset(
        num_samples=10,
        min_steps=3,
        max_steps=10
    )
    
    # 统计
    valid_count = 0
    for sample in dataset:
        # 从question提取relations来验证
        lines = sample['question'].split('\n')
        premise_lines = lines[:-1]  # 除去最后的问题行
        
        # 简单验证：问题中的查询实体不应该在前提中有直接关系
        query_line = lines[-1]  # "Where is X relative to Y?"
        
        # 检查query_entities
        query_entities = sample.get('query_entities', [])
        if len(query_entities) == 2:
            # 检查前提中是否包含这两个实体的直接关系
            has_direct = False
            for premise in premise_lines:
                if all(entity in premise for entity in query_entities):
                    # 两个实体都在同一行，说明有直接关系
                    has_direct = True
                    break
            
            if not has_direct:
                valid_count += 1
    
    print(f"\nResults:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Valid multi-hop reasoning samples: {valid_count}")
    print(f"  Success rate: {valid_count/len(dataset)*100:.1f}%")
    
    # 打印第一个样本
    if dataset:
        print(f"\n{'='*70}")
        print("Sample from generated dataset:")
        print('='*70)
        generator.print_sample(dataset[0])

if __name__ == "__main__":
    print("Testing Spatial Reasoning Generator")
    print("="*70)
    
    # 测试单个样本
    test_single_sample()
    
    # 测试批量生成
    test_batch_generation()
    
    print(f"\n{'='*70}")
    print("Testing complete!")
    print('='*70)



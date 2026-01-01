#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多跳空间推理数据集生成器
使用大写字母作为实体，支持6个方向的空间关系推理
"""

import json
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SpatialRelation:
    """空间关系"""
    subject: str
    relation: str
    object: str
    
    def __str__(self):
        return f"{self.subject}在{self.object}的{self.relation}。"


class SpatialReasoningGenerator:
    """空间推理数据集生成器"""
    
    # 定义6个基本方向（中文）
    RELATIONS = ['左边', '右边', '前面', '后面', '上面', '下面']
    
    # 定义反向关系
    OPPOSITE_RELATIONS = {
        '左边': '右边',
        '右边': '左边',
        '前面': '后面',
        '后面': '前面',
        '上面': '下面',
        '下面': '上面'
    }
    
    # 定义3D空间向量表示 (x, y, z)
    # x轴: 左边(-) / 右边(+)
    # y轴: 后面(-) / 前面(+)
    # z轴: 下面(-) / 上面(+)
    RELATION_VECTORS = {
        '左边': np.array([-1, 0, 0]),
        '右边': np.array([1, 0, 0]),
        '前面': np.array([0, 1, 0]),
        '后面': np.array([0, -1, 0]),
        '上面': np.array([0, 0, 1]),
        '下面': np.array([0, 0, -1])
    }
    
    # 中文序数（天干 + 地支开头）
    CHINESE_ENTITIES = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸', '子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉']
    
    def __init__(self, seed: Optional[int] = None):
        """初始化生成器"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def vector_to_relation(self, vector: np.ndarray) -> Optional[str]:
        """将向量转换为空间关系描述"""
        x, y, z = vector
        
        # 如果向量为零，返回None
        if x == 0 and y == 0 and z == 0:
            return "相同位置"
        
        # 构建复合关系描述
        relations = []
        
        # Z轴 (垂直方向)
        if z > 0:
            if z == 1:
                relations.append("上面")
            else:
                relations.append(f"上面（{z}步）")
        elif z < 0:
            if z == -1:
                relations.append("下面")
            else:
                relations.append(f"下面（{-z}步）")
        
        # Y轴 (前后方向)
        if y > 0:
            if y == 1:
                relations.append("前面")
            else:
                relations.append(f"前面（{y}步）")
        elif y < 0:
            if y == -1:
                relations.append("后面")
            else:
                relations.append(f"后面（{-y}步）")
        
        # X轴 (左右方向)
        if x > 0:
            if x == 1:
                relations.append("右边")
            else:
                relations.append(f"右边（{x}步）")
        elif x < 0:
            if x == -1:
                relations.append("左边")
            else:
                relations.append(f"左边（{-x}步）")
        
        # 组合关系
        if len(relations) == 1:
            return relations[0]
        else:
            return "且".join(relations)
    
    def generate_chain(self, num_steps: int) -> Tuple[List[SpatialRelation], Dict[str, np.ndarray], str, str]:
        """
        生成一条空间关系链，确保需要多跳推理
        
        Args:
            num_steps: 推理步骤数量 (3-10)
            
        Returns:
            relations: 空间关系列表
            positions: 各实体的位置字典
            start_entity: 起始实体
            end_entity: 终止实体
        """
        # 需要 num_steps + 1 个实体来形成一条链
        num_entities = num_steps + 1
        entities = self.CHINESE_ENTITIES[:num_entities]  # 甲, 乙, 丙, 丁, ...
        
        # 初始化位置，第一个实体在原点
        positions = {entities[0]: np.array([0, 0, 0])}
        
        # 生成严格的链式结构：A -> B -> C -> D -> ...
        # 确保每个新实体只参照链中前面的某个实体（但不能是最后一个实体，以保证多跳推理）
        relations = []
        
        for i in range(num_steps):
            # 当前实体
            current_entity = entities[i + 1]
            
            # 选择参照实体：从链的起点到当前位置之前的所有实体中选择
            # 但如果是最后一个实体，不能直接参照起点（确保需要推理）
            if i == num_steps - 1 and num_steps > 1:
                # 最后一个实体：只能参照中间的实体，不能直接参照起点
                possible_references = entities[1:i+1]
            else:
                # 其他实体：可以参照之前的任何实体
                possible_references = entities[:i+1]
            
            reference_entity = random.choice(possible_references)
            
            # 随机选择一个方向
            relation = random.choice(self.RELATIONS)
            
            # 计算新实体的位置
            # 如果 A is left of B，则 A的位置 = B的位置 + left向量
            positions[current_entity] = positions[reference_entity] + self.RELATION_VECTORS[relation]
            
            # 添加关系
            relations.append(SpatialRelation(
                subject=current_entity,
                relation=relation,
                object=reference_entity
            ))
        
        # 问题总是询问第一个和最后一个实体的关系
        start_entity = entities[0]
        end_entity = entities[-1]
        
        return relations, positions, start_entity, end_entity
    
    def verify_needs_reasoning(
        self,
        relations: List[SpatialRelation],
        start_entity: str,
        end_entity: str
    ) -> bool:
        """
        验证问题是否需要多跳推理（起点和终点之间没有直接关系）
        
        Args:
            relations: 空间关系列表
            start_entity: 起始实体
            end_entity: 终止实体
            
        Returns:
            True if the question requires multi-hop reasoning
        """
        for rel in relations:
            # 检查是否存在直接关系
            if (rel.subject == start_entity and rel.object == end_entity) or \
               (rel.subject == end_entity and rel.object == start_entity):
                return False
        return True
    
    def generate_question_answer(
        self, 
        relations: List[SpatialRelation], 
        positions: Dict[str, np.ndarray],
        start_entity: str,
        end_entity: str
    ) -> Tuple[str, str, np.ndarray]:
        """
        生成问题和答案
        
        Args:
            relations: 空间关系列表
            positions: 各实体的位置字典
            start_entity: 起始实体
            end_entity: 终止实体
            
        Returns:
            question: 问题字符串
            answer: 答案字符串
            target: 目标向量 (probe坐标系: [x, y, z] = [left/right, below/above, behind/front])
        """
        # 随机决定问哪个方向
        if random.random() < 0.5:
            query_subject = end_entity
            query_object = start_entity
        else:
            query_subject = start_entity
            query_object = end_entity
        
        # 构建问题
        premise = "\n".join([str(rel) for rel in relations])
        question = f"{premise}\n{query_subject}在{query_object}的什么位置？"
        
        # 计算答案
        # 在生成器坐标系中: [x, y, z] = [left/right, behind/front, below/above]
        relative_vector = positions[query_subject] - positions[query_object]
        answer = self.vector_to_relation(relative_vector)
        
        # 转换到probe坐标系: [x, y, z] = [left/right, below/above, behind/front]
        # target = [gen_x, gen_z, gen_y]
        target = np.array([relative_vector[0], relative_vector[2], relative_vector[1]], dtype=np.float32)
        
        return question, answer, target
    
    def generate_dataset(
        self, 
        num_samples: int = 1000,
        min_steps: int = 3,
        max_steps: int = 10,
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """
        生成完整的数据集
        
        Args:
            num_samples: 样本数量
            min_steps: 最小推理步骤
            max_steps: 最大推理步骤
            output_file: 输出文件路径（可选）
            
        Returns:
            dataset: 数据集列表
        """
        dataset = []
        
        for i in range(num_samples):
            # 随机选择推理步骤数
            num_steps = random.randint(min_steps, max_steps)
            
            # 生成关系链，确保需要多跳推理
            max_attempts = 10
            for attempt in range(max_attempts):
                relations, positions, start_entity, end_entity = self.generate_chain(num_steps)
                
                # 验证是否需要推理
                if self.verify_needs_reasoning(relations, start_entity, end_entity):
                    break
            
            # 生成问题和答案
            question, answer, target = self.generate_question_answer(relations, positions, start_entity, end_entity)
            
            # 构建数据样本
            sample = {
                'id': i + 1,
                'num_steps': num_steps,
                'question': question,
                'answer': answer,
                'target': target.tolist(),
                'entities': list(positions.keys()),
                'positions': {k: v.tolist() for k, v in positions.items()},
                'query_entities': [start_entity, end_entity]
            }
            
            dataset.append(sample)
            
            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1}/{num_samples} 个样本...")
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"\n数据集已保存至 {output_file}")
        
        return dataset
    
    def print_sample(self, sample: Dict):
        """打印样本示例"""
        print(f"\n{'='*60}")
        print(f"样本ID: {sample['id']}")
        print(f"推理步骤数: {sample['num_steps']}")
        print(f"\n{sample['question']}")
        print(f"\n答案: {sample['answer']}")
        print(f"{'='*60}\n")


def main():
    """主函数"""
    # 创建生成器
    generator = SpatialReasoningGenerator(seed=36)#42
    
    # 生成数据集
    print("正在生成中文空间推理数据集...")
    dataset = generator.generate_dataset(
        num_samples=100, #1000
        min_steps=3,
        max_steps=10,
        output_file='spatial_reasoning_dataset_ZH_test.json' #
    )
    
    # 打印一些示例
    print("\n" + "="*60)
    print("样本示例:")
    print("="*60)
    
    # 打印不同推理步骤的示例
    for num_steps in [3, 5, 8, 10]:
        samples_with_n_steps = [s for s in dataset if s['num_steps'] == num_steps]
        if samples_with_n_steps:
            generator.print_sample(samples_with_n_steps[0])
    
    # 统计信息
    print("\n" + "="*60)
    print("数据集统计:")
    print("="*60)
    print(f"总样本数: {len(dataset)}")
    
    step_distribution = defaultdict(int)
    for sample in dataset:
        step_distribution[sample['num_steps']] += 1
    
    print("\n按步骤数分布:")
    for steps in sorted(step_distribution.keys()):
        print(f"  {steps}步: {step_distribution[steps]} 个样本")


if __name__ == "__main__":
    main()


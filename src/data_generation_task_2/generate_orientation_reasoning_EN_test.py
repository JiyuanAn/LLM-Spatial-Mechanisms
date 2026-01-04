#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向推理测试数据集生成器（小规模测试集）
"""

import json
import random
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TurnAction:
    """转向动作"""
    action: str
    
    def __str__(self):
        return f"{self.action}."


class OrientationReasoningGenerator:
    """方向推理数据集生成器"""
    
    # 定义4个基本方向（顺时针顺序）
    DIRECTIONS = ['north', 'east', 'south', 'west']
    
    # 定义转向动作
    ACTIONS = ['Turn right', 'Turn left', 'Turn around']
    
    # 定义动作对应的旋转角度（以90度为单位，顺时针为正）
    ACTION_ROTATIONS = {
        'Turn right': 1,      # 顺时针90度
        'Turn left': -1,      # 逆时针90度
        'Turn around': 2      # 180度
    }
    
    # 定义方向对应的角度（度数，从东开始逆时针）
    # 东: 0°, 北: 90°, 西: 180°, 南: 270°
    DIRECTION_ANGLES = {
        'east': 0,
        'north': 90,
        'west': 180,
        'south': 270
    }
    
    def __init__(self, seed: Optional[int] = None):
        """初始化生成器"""
        if seed is not None:
            random.seed(seed)
    
    def encode_direction(self, direction: str) -> List[float]:
        """
        将方向编码为 [cos(θ), sin(θ)]
        
        Args:
            direction: 方向字符串 (north/east/south/west)
            
        Returns:
            encoded: [cos(θ), sin(θ)] 编码
        """
        angle_degrees = self.DIRECTION_ANGLES[direction]
        angle_radians = math.radians(angle_degrees)
        
        cos_val = math.cos(angle_radians)
        sin_val = math.sin(angle_radians)
        
        # 处理浮点数精度问题，四舍五入到小数点后10位
        cos_val = round(cos_val, 10)
        sin_val = round(sin_val, 10)
        
        return [cos_val, sin_val]
    
    def apply_action(self, current_direction: str, action: str) -> str:
        """应用转向动作"""
        current_index = self.DIRECTIONS.index(current_direction)
        rotation = self.ACTION_ROTATIONS[action]
        new_index = (current_index + rotation) % 4
        return self.DIRECTIONS[new_index]
    
    def generate_action_sequence(self, num_steps: int) -> Tuple[List[TurnAction], str, str]:
        """生成转向动作序列"""
        start_direction = random.choice(self.DIRECTIONS)
        current_direction = start_direction
        
        actions = []
        for _ in range(num_steps):
            action = random.choice(self.ACTIONS)
            actions.append(TurnAction(action=action))
            current_direction = self.apply_action(current_direction, action)
        
        end_direction = current_direction
        return actions, start_direction, end_direction
    
    def verify_needs_reasoning(self, actions: List[TurnAction]) -> bool:
        """验证问题是否需要多步推理"""
        return len(actions) > 1
    
    def generate_distractor_options(self, correct_answer: str) -> List[str]:
        """生成干扰选项"""
        distractors = [d for d in self.DIRECTIONS if d != correct_answer]
        return distractors
    
    def generate_question_answer(
        self, 
        actions: List[TurnAction],
        start_direction: str,
        end_direction: str
    ) -> Tuple[str, str, List[str], str]:
        """生成问题和答案"""
        premise_lines = [f"You are facing {start_direction}."]
        premise_lines.extend([str(action) for action in actions])
        premise = "\n".join(premise_lines)
        question = f"{premise}\n\nWhich direction are you facing now?"
        
        answer = end_direction
        distractors = self.generate_distractor_options(answer)
        all_options = [answer] + distractors
        random.shuffle(all_options)
        
        correct_index = all_options.index(answer)
        correct_option = chr(65 + correct_index)
        
        return question, answer, all_options, correct_option
    
    def generate_dataset(
        self, 
        num_samples: int = 100,
        min_steps: int = 2,
        max_steps: int = 15,
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """生成完整的数据集"""
        dataset = []
        
        for i in range(num_samples):
            num_steps = random.randint(min_steps, max_steps)
            actions, start_direction, end_direction = self.generate_action_sequence(num_steps)
            
            if not self.verify_needs_reasoning(actions):
                continue
            
            question, answer, options, correct_option = self.generate_question_answer(
                actions, start_direction, end_direction
            )
            
            # 编码最终方向为 [cos(θ), sin(θ)]
            target_encoding = self.encode_direction(end_direction)
            
            sample = {
                'id': i + 1,
                'num_steps': num_steps,
                'question': question,
                'answer': answer,
                'options': options,
                'correct_option': correct_option,
                'start_direction': start_direction,
                'end_direction': end_direction,
                'actions': [action.action for action in actions],
                'target': target_encoding
            }
            
            dataset.append(sample)
            
            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1}/{num_samples} samples...")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"\nDataset saved to {output_file}")
        
        return dataset
    
    def print_sample(self, sample: Dict):
        """打印样本示例"""
        print(f"\n{'='*60}")
        print(f"Sample ID: {sample['id']}")
        print(f"Number of reasoning steps: {sample['num_steps']}")
        print(f"\n{sample['question']}")
        print(f"\nOptions:")
        for i, option in enumerate(sample['options']):
            option_label = chr(65 + i)  # A, B, C, D
            mark = " ✓" if option_label == sample['correct_option'] else ""
            print(f"  {option_label}. {option}{mark}")
        print(f"\nCorrect Answer: {sample['correct_option']} ({sample['answer']})")
        print(f"Start: {sample['start_direction']} -> End: {sample['end_direction']}")
        print(f"Target Encoding [cos(θ), sin(θ)]: {sample['target']}")
        print(f"{'='*60}\n")


def main():
    """主函数"""
    generator = OrientationReasoningGenerator(seed=36)
    
    print("Generating orientation reasoning test dataset...")
    dataset = generator.generate_dataset(
        num_samples=100,
        min_steps=2,
        max_steps=5,
        output_file='orientation_reasoning_dataset_EN_test.json'
    )
    
    print("\n" + "="*60)
    print("Sample Examples:")
    print("="*60)
    
    for num_steps in [2, 5]:
        samples_with_n_steps = [s for s in dataset if s['num_steps'] == num_steps]
        if samples_with_n_steps:
            generator.print_sample(samples_with_n_steps[0])
    
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    
    step_distribution = defaultdict(int)
    for sample in dataset:
        step_distribution[sample['num_steps']] += 1
    
    print("\nDistribution by number of steps:")
    for steps in sorted(step_distribution.keys()):
        print(f"  {steps} steps: {step_distribution[steps]} samples")


if __name__ == "__main__":
    main()


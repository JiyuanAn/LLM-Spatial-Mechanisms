#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向推理任务的API测试脚本
测试大语言模型在方向推理任务上的表现
"""

from openai import OpenAI
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

MODEL_NAME = "Qwen2.5-7B-Instruct"

# 初始化模型
client = OpenAI(
    base_url='http://202.112.194.78:8010/v1',
    api_key='EMPTY'
)

def extract_answer(response_text):
    """从模型响应中提取答案"""
    # 尝试多种模式匹配答案
    patterns = [
        r'Answer:\s*([A-D])',
        r'answer:\s*([A-D])',
        r'^([A-D])\.',
        r'选择\s*([A-D])',
        r'答案是\s*([A-D])',
        r'答案：\s*([A-D])',
        r'\b([A-D])\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    return None

def call_api(prompt):
    """调用API"""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=500
    )
    return completion.choices[0].message.content

def process_one_item(item):
    """处理单个数据项"""
    try:
        # 调用API
        response = call_api(item['prompt'])
        
        # 提取答案
        predicted_answer = extract_answer(response)
        
        # 判断是否正确
        is_correct = (predicted_answer == item['correct_option'])
        
        # 返回结果
        return {
            'id': item['id'],
            'num_steps': item['num_steps'],
            'correct_option': item['correct_option'],
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'response_text': response,
            'question': item['question'],
            'success': True
        }
    except Exception as e:
        return {
            'id': item.get('id', 'unknown'),
            'num_steps': item.get('num_steps', 0),
            'correct_option': item.get('correct_option', ''),
            'predicted_answer': None,
            'is_correct': False,
            'response_text': f"Error: {str(e)}",
            'question': item.get('question', ''),
            'error': str(e),
            'success': False
        }

if __name__ == "__main__":
    # 读取 JSON 数据
    json_path = os.path.join(os.path.dirname(__file__), 'orientation_reasoning_dataset_EN_test_with_prompt.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items from {json_path}")
    
    # 并行处理
    max_workers = int(os.getenv("API_TEST_MAX_WORKERS", "32"))
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_item, item) for item in data]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            results.append(result)
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    correct_count = sum(1 for r in results if r['success'] and r['is_correct'])
    
    # 按步骤数统计准确率
    step_accuracy = {}
    for result in results:
        if result['success']:
            num_steps = result['num_steps']
            if num_steps not in step_accuracy:
                step_accuracy[num_steps] = {'correct': 0, 'total': 0}
            step_accuracy[num_steps]['total'] += 1
            if result['is_correct']:
                step_accuracy[num_steps]['correct'] += 1
    
    # 计算每个步骤数的准确率
    for num_steps in step_accuracy:
        stats = step_accuracy[num_steps]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    print(f"\nResults:")
    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count / success_count * 100:.2f}%")
    
    print("\nAccuracy by number of steps:")
    for num_steps in sorted(step_accuracy.keys()):
        stats = step_accuracy[num_steps]
        print(f"  {num_steps} steps: {stats['accuracy']:.2%} "
              f"({stats['correct']}/{stats['total']})")
    
    # 保存结果
    output_data = {
        'model': MODEL_NAME,
        'total_samples': len(results),
        'success_count': success_count,
        'correct_count': correct_count,
        'overall_accuracy': correct_count / success_count if success_count > 0 else 0,
        'step_accuracy': step_accuracy,
        'results': results
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'orientation_test_results_test.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")


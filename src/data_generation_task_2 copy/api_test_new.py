from openai import OpenAI
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

MODEL_NAME = "Qwen2.5-7B-Instruct"

# 初始化模型
client = OpenAI(
    base_url='http://202.112.194.78:8010/v1',
    api_key='EMPTY'
)

system_prompt = """"""

def call_api(input_text):
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
    )
    return completion.choices[0].message.content

def process_one_item(item):
    """处理单个数据项"""
    try:
        # 解析 question 字段
        prompt = item['prompt']
        print(prompt)
        
        # 调用 API
        response = call_api(prompt)
        print(response)
        print(item['correct_option'])
        
        # 返回结果
        return {
            'id': item['id'],
            'correct_option': item['correct_option'],
            'predicted_option': response.strip(),
            'answer': item['answer'],
            'success': True
        }
    except Exception as e:
        return {
            'id': item.get('id', 'unknown'),
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
    correct_count = sum(1 for r in results if r['success'] and r['predicted_option'] == r['correct_option'])
    
    print(f"\nResults:")
    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count / success_count * 100:.2f}%")
    
    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), 'api_test_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")
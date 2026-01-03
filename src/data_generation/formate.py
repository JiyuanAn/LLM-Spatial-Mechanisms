import json
import os

# 从 api_test_new.py 中提取的 instruction 模板
instruction = """You are given several statements describing the relative positions of objects in a 3D space.
Each statement describes a relative position with EXACTLY ONE UNIT of distance along a single axis. (Objects may occupy the same position in space.)

Statements:
{statements}

Question:
{question}

Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

Instruction:
Output ONLY the letter of the correct option (A, B, C, or D).
Do NOT provide any explanation, reasoning steps, or additional text.
"""

def parse_question(question_text):
    """从 question 字段中解析出 statements 和 question"""
    lines = question_text.strip().split('\n')
    statements_lines = []
    question_line = ""
    
    for line in lines:
        if line.startswith("Where is"):
            question_line = line
        else:
            statements_lines.append(line)
    
    statements = '\n'.join(statements_lines)
    return statements, question_line

def add_prompt_to_item(item):
    """为单个数据项添加 prompt 字段"""
    # 解析 question 字段
    statements, question = parse_question(item['question'])
    
    # 获取选项
    options = item['options']
    option_A = options[0]
    option_B = options[1]
    option_C = options[2]
    option_D = options[3]
    
    # 构造 prompt
    prompt = instruction.format(
        statements=statements, 
        question=question, 
        option_A=option_A, 
        option_B=option_B, 
        option_C=option_C, 
        option_D=option_D
    )
    
    # 添加 prompt 字段到 item
    item['prompt'] = prompt
    
    return item

if __name__ == "__main__":
    # 读取原始 JSON 数据
    input_path = os.path.join(os.path.dirname(__file__), 'spatial_reasoning_dataset_EN_test.json')
    print(f"Reading from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items")
    
    # 为每个数据项添加 prompt 字段
    for item in data:
        add_prompt_to_item(item)
    
    # 保存到新文件
    output_path = os.path.join(os.path.dirname(__file__), 'spatial_reasoning_dataset_EN_test_with_prompt.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(data)} items with prompt field to: {output_path}")
    
    # 打印第一个示例
    print("\n" + "="*50)
    print("Example of the first item's prompt:")
    print("="*50)
    print(data[0]['prompt'])


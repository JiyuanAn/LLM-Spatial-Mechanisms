# Orientation Reasoning Dataset Generator

方向推理数据集生成器 - 用于评测大语言模型的空间方向推理能力

## 概述

本工具生成基于转向操作的方向推理问题，用于评测大语言模型在方向推理任务上的表现。问题格式如下：

```
You are facing north.
Turn right.
Turn left.
Turn around.
Turn right.

Which direction are you facing now?
```

## 文件说明

### 数据生成脚本

1. **generate_orientation_reasoning_EN.py**
   - 生成完整的方向推理数据集（1000个样本）
   - 支持2-15步的推理链
   - 包含四个方向：north, east, south, west
   - 包含三种转向操作：Turn right, Turn left, Turn around

2. **generate_orientation_reasoning_EN_test.py**
   - 生成测试集（100个样本）
   - 用于快速测试和验证

3. **add_prompt_orientation.py**
   - 为数据集添加提示词
   - 将数据转换为适合API调用的格式

### 测试和分析脚本

4. **api_test_orientation.py**
   - API测试脚本
   - 支持OpenAI API及兼容接口
   - 自动评估模型性能

5. **analyze_orientation_results.py**
   - 结果分析脚本
   - 生成准确率统计
   - 绘制可视化图表
   - 支持与空间推理任务对比

## 使用方法

### 1. 生成数据集

```bash
# 生成完整数据集（1000个样本）
python generate_orientation_reasoning_EN.py

# 生成测试集（100个样本）
python generate_orientation_reasoning_EN_test.py
```

生成的文件：
- `orientation_reasoning_dataset_EN.json` - 完整数据集
- `orientation_reasoning_dataset_EN_test.json` - 测试集

### 2. 添加提示词

```bash
# 为完整数据集添加提示词
python add_prompt_orientation.py

# 为测试集添加提示词（使用Python命令）
python -c "
import json

def add_prompt_to_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    prompt_template = '''You are an AI assistant that helps solve orientation reasoning problems.

Please carefully read the following problem and choose the correct answer from the options provided.

Problem:
{question}

Options:
{options_text}

Please provide your answer in the following format:
Answer: [Your choice (A/B/C/D)]
Explanation: [Brief explanation of your reasoning]'''
    
    for sample in dataset:
        options_text = '\\n'.join([
            f\"{chr(65 + i)}. {option}\" 
            for i, option in enumerate(sample['options'])
        ])
        
        prompt = prompt_template.format(
            question=sample['question'],
            options_text=options_text
        )
        
        sample['prompt'] = prompt
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f'Added prompts to {len(dataset)} samples')
    print(f'Dataset with prompts saved to {output_file}')

add_prompt_to_dataset('orientation_reasoning_dataset_EN_test.json', 'orientation_reasoning_dataset_EN_test_with_prompt.json')
"
```

生成的文件：
- `orientation_reasoning_dataset_EN_with_prompt.json`
- `orientation_reasoning_dataset_EN_test_with_prompt.json`

### 3. 测试模型

编辑 `api_test_orientation.py`，配置API密钥和模型：

```python
API_KEY = "your-api-key-here"  # 替换为你的API密钥
BASE_URL = None  # 如果使用兼容的API服务，设置基础URL
MODEL = "gpt-4"  # 模型名称
```

运行测试：

```bash
python api_test_orientation.py
```

生成的文件：
- `orientation_test_results.json` - 测试结果

### 4. 分析结果

```bash
# 仅分析方向推理结果
python analyze_orientation_results.py orientation_test_results.json

# 对比方向推理和空间推理结果
python analyze_orientation_results.py orientation_test_results.json api_test_results.json
```

生成的文件：
- `orientation_accuracy_by_steps.png` - 按步骤数的准确率图表
- `orientation_vs_spatial_comparison.png` - 对比图表（如果提供了空间推理结果）

## 数据格式

### 基础数据格式

```json
{
  "id": 1,
  "num_steps": 5,
  "question": "You are facing north.\nTurn right.\nTurn left.\nTurn around.\nTurn right.\n\nWhich direction are you facing now?",
  "answer": "west",
  "options": ["north", "east", "west", "south"],
  "correct_option": "C",
  "start_direction": "north",
  "end_direction": "west",
  "actions": ["Turn right", "Turn left", "Turn around", "Turn right"]
}
```

### 带提示词的数据格式

在基础格式上增加 `prompt` 字段，包含完整的提示词模板。

### 测试结果格式

```json
{
  "model": "gpt-4",
  "total_samples": 100,
  "correct_count": 85,
  "overall_accuracy": 0.85,
  "step_accuracy": {
    "2": {"correct": 10, "total": 10, "accuracy": 1.0},
    "3": {"correct": 9, "total": 10, "accuracy": 0.9},
    ...
  },
  "results": [
    {
      "id": 1,
      "num_steps": 5,
      "correct_option": "C",
      "predicted_answer": "C",
      "is_correct": true,
      "response_text": "...",
      "question": "..."
    },
    ...
  ]
}
```

## 任务特点

### 方向系统

- **四个基本方向**：north (北), east (东), south (南), west (西)
- **顺时针顺序**：north → east → south → west

### 转向操作

1. **Turn right** - 顺时针旋转90度
2. **Turn left** - 逆时针旋转90度
3. **Turn around** - 旋转180度

### 推理难度

- **最小步骤数**：2步
- **最大步骤数**：15步
- **难度递增**：步骤数越多，推理难度越大

## 与空间推理的区别

| 特性 | 方向推理 | 空间推理 |
|------|---------|---------|
| 问题类型 | 转向操作 | 空间位置关系 |
| 方向数量 | 4个（N/E/S/W） | 6个（left/right/front/behind/above/below） |
| 操作类型 | Turn right/left/around | 位置描述（A is left of B） |
| 推理维度 | 2D（平面） | 3D（空间） |
| 答案类型 | 单一方向 | 可能是组合方向 |

## 示例问题

### 简单示例（2步）

```
You are facing north.
Turn around.
Turn around.

Which direction are you facing now?

Options:
A. east
B. south
C. west
D. north ✓

Correct Answer: D (north)
```

### 中等示例（5步）

```
You are facing east.
Turn around.
Turn around.
Turn around.
Turn left.
Turn right.

Which direction are you facing now?

Options:
A. east
B. west ✓
C. north
D. south

Correct Answer: B (west)
```

### 困难示例（12步）

```
You are facing north.
Turn right.
Turn around.
Turn left.
Turn right.
Turn right.
Turn right.
Turn around.
Turn right.
Turn around.
Turn around.
Turn around.
Turn right.

Which direction are you facing now?

Options:
A. north
B. east
C. west ✓
D. south

Correct Answer: C (west)
```

## 评测指标

1. **总体准确率**：所有样本的正确率
2. **按步骤数准确率**：不同推理步骤数的准确率分布
3. **错误分析**：分析模型在哪些情况下容易出错

## 依赖环境

```bash
pip install openai matplotlib numpy
```

## 注意事项

1. **API密钥**：使用API测试时需要配置有效的API密钥
2. **API限制**：注意API调用频率限制，可调整延迟参数
3. **成本控制**：测试大量样本时注意API调用成本
4. **随机种子**：数据生成使用固定种子（42）以确保可重复性

## 扩展建议

1. **增加复杂度**：
   - 支持更多方向（如8方向或16方向）
   - 增加转向角度的多样性
   - 引入相对转向（如"转向前方的左侧"）

2. **多模态扩展**：
   - 添加地图可视化
   - 生成配图说明
   - 支持语音输入的转向指令

3. **实际应用**：
   - 导航系统评测
   - 机器人指令理解
   - 地图应用的方向推理

## 许可证

本工具作为空间能力评测Benchmark的一部分，用于学术研究和模型评测。

## 联系方式

如有问题或建议，请联系项目维护者。


# Spatial Procedure Execution Dataset Generator

空间过程执行数据集生成器 - 用于评测大语言模型的空间操作执行能力

## 概述

本工具生成基于空间操作序列的推理问题，用于评测大语言模型在空间过程执行任务上的表现。问题格式如下：

```
Start at (0, 0, 0).
Move up by 2 units.
Move forward by 3 units.
Reflect the position across the y-axis.

What is the final position?
```

## 文件说明

### 数据生成脚本

1. **generate_spatial_procedure_EN.py**
   - 生成完整的空间过程执行数据集（1000个样本）
   - 支持2-10步的操作序列
   - 包含五种空间操作：移动、反射、旋转、缩放、平移

2. **generate_spatial_procedure_EN_test.py**
   - 生成测试集（100个样本）
   - 用于快速测试和验证

3. **add_prompt_spatial_procedure.py**
   - 为数据集添加提示词
   - 将数据转换为适合API调用的格式

## 使用方法

### 1. 生成数据集

```bash
# 生成完整数据集（1000个样本）
cd /home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation_task_3
python generate_spatial_procedure_EN.py

# 生成测试集（100个样本）
python generate_spatial_procedure_EN_test.py
```

生成的文件：
- `spatial_procedure_dataset_EN.json` - 完整数据集
- `spatial_procedure_dataset_EN_test.json` - 测试集

### 2. 添加提示词

```bash
# 为数据集添加提示词
python add_prompt_spatial_procedure.py
```

生成的文件：
- `spatial_procedure_dataset_EN_with_prompt.json`
- `spatial_procedure_dataset_EN_test_with_prompt.json`

## 空间操作类型

### 1. 移动 (Move)

在指定方向上移动指定单位数：

- **up**: z轴正方向
- **down**: z轴负方向
- **forward**: y轴正方向
- **backward**: y轴负方向
- **left**: x轴负方向
- **right**: x轴正方向

示例：`Move up by 2 units.` → (0, 0, 0) → (0, 0, 2)

### 2. 反射 (Reflect)

沿指定轴反射位置：

- **x-axis**: y坐标取反
- **y-axis**: x坐标取反
- **z-axis**: z坐标取反

示例：`Reflect the position across the y-axis.` → (2, 3, 4) → (-2, 3, 4)

### 3. 旋转 (Rotate)

绕指定轴旋转指定角度（90°、180°、270°）：

- **x-axis**: y和z坐标变化
- **y-axis**: x和z坐标变化
- **z-axis**: x和y坐标变化

示例：`Rotate 90° around the z-axis.` → (1, 0, 0) → (0, 1, 0)

### 4. 缩放 (Scale)

所有坐标乘以指定因子（2、3、0.5、-1）：

示例：`Scale all coordinates by 2.` → (1, 2, 3) → (2, 4, 6)

### 5. 平移 (Translate)

按指定向量平移：

示例：`Translate by (1, 2, 3).` → (0, 0, 0) → (1, 2, 3)

## 数据格式

### 基础数据格式

```json
{
  "id": 1,
  "num_steps": 4,
  "question": "Start at (0, 0, 0).\nMove up by 2 units.\nMove forward by 3 units.\nReflect the position across the y-axis.\n\nWhat is the final position?",
  "answer": "(0, 3, 2)",
  "options": ["(0, 3, 2)", "(0, -3, 2)", "(0, 3, -2)", "(3, 0, 2)"],
  "correct_option": "A",
  "start_position": [0, 0, 0],
  "final_position": [0, 3, 2],
  "actions": [
    {
      "type": "move",
      "params": {"direction": "up", "units": 2},
      "description": "Move up by 2 units."
    },
    {
      "type": "move",
      "params": {"direction": "forward", "units": 3},
      "description": "Move forward by 3 units."
    },
    {
      "type": "reflect",
      "params": {"axis": "y"},
      "description": "Reflect the position across the y-axis."
    }
  ],
  "target": [0, 3, 2]
}
```

### 字段说明

- **id**: 样本唯一标识符
- **num_steps**: 操作步骤数
- **question**: 问题描述
- **answer**: 正确答案
- **options**: 四个选项（包含正确答案和干扰项）
- **correct_option**: 正确选项标识（A/B/C/D）
- **start_position**: 起始位置（始终为原点）
- **final_position**: 最终位置
- **actions**: 操作序列详细信息
- **target**: 目标向量（用于探针分析）

### 带提示词的数据格式

在基础格式上增加 `prompt` 字段，包含完整的提示词模板。

## 任务特点

### 坐标系统

- **3D笛卡尔坐标系**：(x, y, z)
- **起始点**：原点 (0, 0, 0)
- **坐标轴方向**：
  - x轴：左（-）右（+）
  - y轴：后（-）前（+）
  - z轴：下（-）上（+）

### 推理难度

- **最小步骤数**：2步
- **最大步骤数**：10步
- **难度递增**：步骤数越多，操作类型越复杂，推理难度越大

### 干扰选项设计

干扰选项通过以下方式生成：
1. 某个坐标符号错误
2. 坐标轴交换
3. 坐标值偏移
4. 坐标值缩放错误

## 示例问题

### 简单示例（2步）

```
Start at (0, 0, 0).
Move up by 2 units.
Move right by 3 units.

What is the final position?

Options:
A. (3, 0, 2) ✓
B. (-3, 0, 2)
C. (3, 0, -2)
D. (0, 3, 2)

Correct Answer: A (3, 0, 2)
```

### 中等示例（5步）

```
Start at (0, 0, 0).
Move forward by 2 units.
Move up by 3 units.
Reflect the position across the x-axis.
Move left by 1 unit.
Scale all coordinates by 2.

What is the final position?

Options:
A. (-2, 4, 6)
B. (-2, -4, 6) ✓
C. (2, -4, 6)
D. (-2, 4, -6)

Correct Answer: B (-2, -4, 6)
```

### 困难示例（10步）

包含多种操作类型的复杂组合，需要仔细跟踪每一步的坐标变化。

## 与其他任务的区别

| 特性 | 空间过程执行 | 方向推理 | 空间推理 |
|------|------------|---------|---------|
| 问题类型 | 空间操作序列 | 转向操作 | 空间位置关系 |
| 输出类型 | 3D坐标 | 方向（4个） | 方向（6个） |
| 操作类型 | 移动/反射/旋转/缩放/平移 | Turn right/left/around | 位置描述 |
| 推理维度 | 3D（xyz坐标） | 2D（平面方向） | 3D（空间方向） |
| 答案类型 | 坐标元组 | 单一方向 | 可能是组合方向 |

## 评测指标

1. **总体准确率**：所有样本的正确率
2. **按步骤数准确率**：不同操作步骤数的准确率分布
3. **按操作类型准确率**：不同操作类型的准确率
4. **错误分析**：分析模型在哪些情况下容易出错

## 依赖环境

```bash
pip install numpy
```

注：基础功能仅依赖Python标准库，分析和可视化可能需要额外的库。

## 扩展建议

1. **增加操作类型**：
   - 对称操作（关于点或平面对称）
   - 投影操作
   - 复合旋转

2. **增加复杂度**：
   - 条件操作（如果x>0则...）
   - 循环操作（重复某操作n次）
   - 参数化操作（移动距离依赖于当前坐标）

3. **多模态扩展**：
   - 3D可视化
   - 动画演示操作过程
   - 交互式验证

4. **实际应用**：
   - 机器人路径规划
   - 3D建模指令理解
   - 游戏角色控制

## 注意事项

1. **浮点数精度**：计算中对浮点数进行了适当的四舍五入处理
2. **坐标系统**：使用右手坐标系
3. **随机种子**：数据生成使用固定种子以确保可重复性
4. **操作顺序**：操作按顺序依次执行，每步依赖前一步的结果

## 许可证

本工具作为空间能力评测Benchmark的一部分，用于学术研究和模型评测。

## 联系方式

如有问题或建议，请联系项目维护者。


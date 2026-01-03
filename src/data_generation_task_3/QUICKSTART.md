# 空间过程执行数据集 - 快速入门指南

## 简介

这是一个用于评测大语言模型空间推理能力的数据集生成工具，专注于**空间过程执行（Spatial Procedure Execution）**任务。

## 快速开始

### 1. 数据已生成完毕

以下文件已经生成并可以直接使用：

```
✓ spatial_procedure_dataset_EN.json                    # 完整数据集（1000样本）
✓ spatial_procedure_dataset_EN_test.json               # 测试集（100样本）
✓ spatial_procedure_dataset_EN_with_prompt.json        # 完整数据集+提示词
✓ spatial_procedure_dataset_EN_test_with_prompt.json   # 测试集+提示词
```

### 2. 查看数据示例

运行示例脚本查看数据集内容：

```bash
python example_usage.py
```

### 3. 重新生成数据（可选）

如果需要重新生成数据集：

```bash
# 生成完整数据集（1000样本）
python generate_spatial_procedure_EN.py

# 生成测试集（100样本）
python generate_spatial_procedure_EN_test.py

# 添加提示词
python add_prompt_spatial_procedure.py
```

### 4. 测试模型性能

修改 `api_test_spatial_procedure.py` 中的API配置：

```python
MODEL_NAME = "your-model-name"
client = OpenAI(
    base_url='your-api-base-url',
    api_key='your-api-key'
)
```

运行测试：

```bash
python api_test_spatial_procedure.py
```

### 5. 分析测试结果

```bash
python analyze_spatial_procedure_results.py spatial_procedure_test_results.json
```

## 任务说明

### 任务类型

空间过程执行任务要求模型：
1. 从起始点 (0, 0, 0) 开始
2. 依次执行一系列空间操作
3. 计算最终位置坐标

### 支持的操作类型

1. **移动 (Move)**: 在6个方向上移动
   - `Move up by 2 units` - 向上移动2个单位
   - `Move right by 3 units` - 向右移动3个单位
   
2. **反射 (Reflect)**: 沿坐标轴反射
   - `Reflect the position across the y-axis` - 沿y轴反射
   
3. **旋转 (Rotate)**: 绕坐标轴旋转
   - `Rotate 90° around the z-axis` - 绕z轴旋转90度
   
4. **缩放 (Scale)**: 坐标缩放
   - `Scale all coordinates by 2` - 所有坐标乘以2
   
5. **平移 (Translate)**: 向量平移
   - `Translate by (1, 2, 3)` - 平移向量(1, 2, 3)

### 示例问题

```
Start at (0, 0, 0).
Move up by 2 units.
Move forward by 3 units.
Reflect the position across the y-axis.

What is the final position?

Options:
A. (0, 3, 2) ✓
B. (0, -3, 2)
C. (3, 0, 2)
D. (0, 3, -2)
```

**解答过程**：
- 起始: (0, 0, 0)
- 向上移动2单位: (0, 0, 2)
- 向前移动3单位: (0, 3, 2)
- 沿y轴反射: (0, 3, 2) → x坐标不变，y坐标不变，z坐标不变
- 最终: (0, 3, 2)

## 数据集统计

### 完整数据集（1000样本）
- 步骤数范围: 2-10步
- 每种步骤数约100个样本
- 五种操作类型均匀分布

### 测试集（100样本）
- 快速验证模型性能
- 步骤数分布与完整数据集一致

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_spatial_procedure_EN.py` | 主要生成脚本（1000样本）|
| `generate_spatial_procedure_EN_test.py` | 测试集生成脚本（100样本）|
| `add_prompt_spatial_procedure.py` | 添加提示词脚本 |
| `api_test_spatial_procedure.py` | API测试脚本 |
| `analyze_spatial_procedure_results.py` | 结果分析脚本 |
| `example_usage.py` | 使用示例脚本 |
| `README_SPATIAL_PROCEDURE.md` | 详细文档 |
| `QUICKSTART.md` | 本快速入门指南 |

## 进阶使用

### 自定义生成参数

修改生成器参数：

```python
from generate_spatial_procedure_EN import SpatialProcedureGenerator

generator = SpatialProcedureGenerator(seed=42)
dataset = generator.generate_dataset(
    num_samples=2000,    # 样本数量
    min_steps=3,         # 最小步骤数
    max_steps=15,        # 最大步骤数
    output_file='custom_dataset.json'
)
```

### 验证数据正确性

使用示例脚本验证：

```python
from example_usage import verify_solution

# 验证单个样本
verify_solution(sample)
```

### 与其他任务对比

分析脚本支持与其他空间推理任务对比：

```bash
python analyze_spatial_procedure_results.py \
    spatial_procedure_test_results.json \
    orientation_test_results.json
```

## 常见问题

### Q: 坐标系统是什么？
A: 使用右手坐标系，x轴（左右），y轴（前后），z轴（上下）

### Q: 为什么有些坐标是 -0.0？
A: 这是浮点数运算的正常现象，-0.0 等价于 0.0

### Q: 如何增加任务难度？
A: 增加 max_steps 参数，或添加更多操作类型

### Q: 干扰选项如何生成？
A: 通过坐标符号错误、坐标轴交换、数值偏移等方式生成

## 相关资源

- 详细文档: `README_SPATIAL_PROCEDURE.md`
- 示例代码: `example_usage.py`
- 项目主文档: `/workspace/SA-of-LLM/Doc/Readme.md`

## 技术支持

如有问题或建议，请参考详细文档或联系项目维护者。


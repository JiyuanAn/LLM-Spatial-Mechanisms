"""
快速诊断脚本：检查 probe 为什么失败
"""
import sys
sys.path.append("./")
sys.path.append("../../")

import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 加载数据
print("="*60)
print("加载 Spatial Procedure 数据...")
print("="*60)

with open('../data_generation_task_3/spatial_procedure_dataset_EN.json', 'r') as f:
    train_data = json.load(f)

with open('../data_generation_task_3/spatial_procedure_dataset_EN_test.json', 'r') as f:
    test_data = json.load(f)

train_targets = np.array([s['target'] for s in train_data])
test_targets = np.array([s['target'] for s in test_data])

print(f"训练集: {len(train_data)} 样本")
print(f"测试集: {len(test_data)} 样本")
print()

# 测试1: 基线模型 - 预测均值
print("="*60)
print("测试 1: 基线模型 - 预测训练集均值")
print("="*60)

train_mean = train_targets.mean(axis=0)
print(f"训练集均值: {train_mean}")

# 用训练集均值预测测试集
baseline_pred = np.tile(train_mean, (len(test_targets), 1))
baseline_r2 = r2_score(test_targets, baseline_pred, multioutput='uniform_average')
baseline_mse = mean_squared_error(test_targets, baseline_pred)

print(f"基线 R²: {baseline_r2:.4f}")
print(f"基线 MSE: {baseline_mse:.4f}")
print()

# 测试2: 用随机特征做 Ridge 回归
print("="*60)
print("测试 2: 使用随机特征的 Ridge 回归")
print("="*60)

np.random.seed(42)
d_model = 3584  # Qwen 模型的维度

# 生成随机特征
X_train_random = np.random.randn(len(train_targets), d_model).astype(np.float32)
X_test_random = np.random.randn(len(test_targets), d_model).astype(np.float32)

# 尝试不同的 alpha 值
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

print(f"特征维度: {d_model}")
print(f"样本数量: {len(train_targets)}")
print(f"维度/样本比: {d_model/len(train_targets):.2f}")
print()

best_alpha = None
best_r2 = -np.inf

for alpha in alphas:
    probe = Ridge(alpha=alpha)
    probe.fit(X_train_random, train_targets)
    y_pred = probe.predict(X_test_random)
    r2 = r2_score(test_targets, y_pred, multioutput='uniform_average')
    mse = mean_squared_error(test_targets, y_pred)
    
    print(f"Alpha={alpha:7.2f} | R²={r2:7.4f} | MSE={mse:7.2f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_alpha = alpha

print()
print(f"最佳 alpha: {best_alpha} (R² = {best_r2:.4f})")
print()

# 测试3: 检查目标值分布
print("="*60)
print("测试 3: 目标值分布分析")
print("="*60)

print("训练集统计:")
for i, dim in enumerate(['x', 'y', 'z']):
    print(f"  {dim}: mean={train_targets[:, i].mean():6.2f}, "
          f"std={train_targets[:, i].std():6.2f}, "
          f"range=[{train_targets[:, i].min():6.2f}, {train_targets[:, i].max():6.2f}]")

print("\n测试集统计:")
for i, dim in enumerate(['x', 'y', 'z']):
    print(f"  {dim}: mean={test_targets[:, i].mean():6.2f}, "
          f"std={test_targets[:, i].std():6.2f}, "
          f"range=[{test_targets[:, i].min():6.2f}, {test_targets[:, i].max():6.2f}]")

print()

# 测试4: 与 Spatial Reasoning 对比
print("="*60)
print("测试 4: 与 Spatial Reasoning 任务对比")
print("="*60)

with open('../data_generation/spatial_reasoning_dataset_EN_with_prompt.json', 'r') as f:
    sr_data = json.load(f)

sr_targets = np.array([s['target'] for s in sr_data[:1000]])

print("Spatial Reasoning 统计:")
for i, dim in enumerate(['x', 'y', 'z']):
    print(f"  {dim}: mean={sr_targets[:, i].mean():6.2f}, "
          f"std={sr_targets[:, i].std():6.2f}, "
          f"range=[{sr_targets[:, i].min():6.2f}, {sr_targets[:, i].max():6.2f}]")

print("\nSpatial Procedure 统计:")
for i, dim in enumerate(['x', 'y', 'z']):
    print(f"  {dim}: mean={train_targets[:, i].mean():6.2f}, "
          f"std={train_targets[:, i].std():6.2f}, "
          f"range=[{train_targets[:, i].min():6.2f}, {train_targets[:, i].max():6.2f}]")

print()
print("="*60)
print("结论和建议")
print("="*60)
print("""
1. 如果随机特征的 R² 接近 0 或为负，说明：
   - 特征维度远大于样本数量（过拟合风险）
   - 需要更多训练数据或更强的正则化

2. 如果模型 R² 比随机特征还差，说明：
   - 模型隐藏状态不包含目标信息
   - 模型可能没有在类似任务上训练过

3. Spatial Procedure 的目标值范围比 Spatial Reasoning 大得多：
   - 可能需要标准化目标值
   - 或者改为预测"位移向量"而非绝对位置
""")


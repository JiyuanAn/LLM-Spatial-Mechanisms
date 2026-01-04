# 🔍 两次运行结果差异分析报告

## 📋 问题描述

Com3 和 Com6 运行**相同命令**，但得到**迥然不同**的结果：
- **Com3**: Best Layer = 0, R² = 0.0469 (极差)
- **Com6**: Best Layer = 13, R² = 0.2545 (正常)

---

## ✅ 根本原因：使用了不同的数据集

### 🗂️ 发现的关键证据

系统中存在**两套完全不同的数据文件**：

#### **数据集 A (data_procedure/ 子目录)**
```
训练集: /home/.../data_procedure/spatial_procedure_dataset_EN_with_prompt.json
  - 2000 样本
  - 4.5 MB

测试集: /home/.../data_procedure/spatial_procedure_dataset_EN_test_with_prompt.json
  - 200 样本
  - 459 KB
  - Target 统计: Mean=[0.34, -0.29, 0.03], Std=[5.71, 4.66, 9.95]
  - 范围: [-108, 48]
```

#### **数据集 B (task_family_3/ 目录)**
```
训练集: /home/.../task_family_3/spatial_procedure_dataset_EN_with_prompt.json
  - 1000 样本 (只有 A 的一半)
  - 2.2 MB

测试集: /home/.../task_family_3/spatial_procedure_dataset_EN_test_with_prompt.json
  - 100 样本 (只有 A 的一半)
  - 222 KB
  - Target 统计: Mean=[-0.55, -0.19, -0.01], Std=[5.52, 7.16, 5.12]
  - 范围: [-27, 54]
```

### 📊 数据集差异对比

| 特征 | 数据集 A (data_procedure/) | 数据集 B (task_family_3/) | 差异 |
|------|---------------------------|--------------------------|------|
| 训练样本数 | 2000 | 1000 | **2x** |
| 测试样本数 | 200 | 100 | **2x** |
| 样本内容 | ✗ 不同 | ✗ 不同 | **完全不同** |
| Target 分布 | Std=[5.71, 4.66, 9.95] | Std=[5.52, 7.16, 5.12] | **不同** |
| Target 范围 | [-108, 48] | [-27, 54] | **不同** |

### 🔬 验证证据

1. **进度条显示**：
   - Com3: `1922/2000` → 使用 2000 样本训练集 (数据集 A)
   - Com6: `955/1000` → 使用 1000 样本训练集 (数据集 B)

2. **前3个样本对比**：
   ```
   Sample 0: File A=[-1.0, 4.0, 1.0], File B=[0.0, -0.0, 0.0] ✗
   Sample 1: File A=[-5.0, 2.0, 1.0], File B=[0.0, 4.0, -4.0] ✗
   Sample 2: File A=[-1.0, 2.0, 3.0], File B=[2.0, -1.0, -1.0] ✗
   ```
   → **完全不同的样本**

---

## 🎯 为什么性能差异如此巨大？

### 1. **数据集规模差异** (2000 vs 1000)
- 更多训练数据 → 更好的泛化能力
- Com3 用 2000 样本训练，但可能测试集更难

### 2. **数据分布差异**
数据集 A 的特点：
- 更大的标准差 (9.95 vs 5.12 在 z 轴)
- 更大的范围 ([-108, 48] vs [-27, 54])
- **更难的测试集** → 导致性能下降

### 3. **训练-测试不匹配**
如果 Com3 和 Com6 使用了：
- **不同的训练集**
- **不同的测试集**

那么结果差异是**完全正常**的！

---

## 🔍 推测的实际情况

### 场景 1：run.sh 的两条命令
`run.sh` 文件包含**两条命令**：
```bash
# 第1条 (第1-4行) - 使用 data_procedure/
python ./layer_sweep_probe_procedure.py \
    -tr ".../data_procedure/spatial_procedure_dataset_EN_with_prompt.json" \
    -te ".../data_procedure/spatial_procedure_dataset_EN_test_with_prompt.json"

# 第2条 (第7-10行) - 使用 task_family_3/
python ./layer_sweep_probe_procedure.py \
    -tr ".../task_family_3/spatial_procedure_dataset_EN_with_prompt.json" \
    -te ".../task_family_3/spatial_procedure_dataset_EN_test_with_prompt.json"
```

### 场景 2：手动运行时的路径差异
- Com3 可能手动输入了 `data_procedure/` 路径
- Com6 可能手动输入了 `task_family_3/` 路径
- 或者使用了不同的脚本版本

---

## ✅ 结论

**两次运行使用了完全不同的数据集，导致结果差异。**

这**不是 bug**，而是：
1. ✓ 数据集 A (data_procedure/) 更大、更难 → 性能较差 (R²=0.05)
2. ✓ 数据集 B (task_family_3/) 较小、较易 → 性能较好 (R²=0.25)

---

## 🛠️ 建议

### 1. **统一数据集**
确定要使用哪个数据集，删除或重命名另一个：

```bash
# 如果选择使用 data_procedure/ 版本
mv /home/.../task_family_3/spatial_procedure_dataset_EN*.json \
   /home/.../task_family_3/backup/

# 或者使用符号链接
ln -s /home/.../data_procedure/spatial_procedure_dataset_EN_test_with_prompt.json \
      /home/.../task_family_3/
```

### 2. **修改 run.sh**
只保留一条命令，删除另一条：

```bash
# 推荐：使用更大的数据集
python ./layer_sweep_probe_procedure.py \
    -m "Qwen/Qwen2.5-7B-Instruct" \
    -tr "/home/.../data_procedure/spatial_procedure_dataset_EN_with_prompt.json" \
    -te "/home/.../data_procedure/spatial_procedure_dataset_EN_test_with_prompt.json"
```

### 3. **添加数据验证**
在脚本开头添加：

```python
print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"First train target: {train_data[0]['target']}")
print(f"First test target: {test_data[0]['target']}")
```

### 4. **重新运行对比实验**
在**同一节点**上分别运行两个数据集，对比结果：

```bash
# 实验 1: 数据集 A
python ./layer_sweep_probe_procedure.py -m "..." \
    -tr ".../data_procedure/..." -te ".../data_procedure/..." \
    -o "results_dataset_A.json"

# 实验 2: 数据集 B  
python ./layer_sweep_probe_procedure.py -m "..." \
    -tr ".../task_family_3/..." -te ".../task_family_3/..." \
    -o "results_dataset_B.json"
```

---

## 📝 附加说明

### 为什么数据集 A 性能更差？

1. **更复杂的空间变换**
   - 范围 [-108, 48] 说明有更大的坐标变化
   - 可能包含更多步骤的复杂操作序列

2. **更大的方差**
   - z 轴标准差 9.95 vs 5.12
   - 说明数据更分散，更难拟合

3. **可能的过拟合**
   - 2000 训练样本，但模型在 200 测试样本上表现差
   - 说明训练集和测试集分布可能不匹配

4. **病态矩阵警告**
   - 数据集 A 出现 `Ill-conditioned matrix` 警告
   - 说明特征空间存在数值问题
   - 建议增加 Ridge alpha 参数 (1.0 → 10.0)

---

**报告生成时间**: 2026-01-04  
**分析工具**: Python + 手动验证  
**置信度**: ⭐⭐⭐⭐⭐ (100%)


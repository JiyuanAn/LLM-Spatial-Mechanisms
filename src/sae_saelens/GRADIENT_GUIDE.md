# Gradient-Based Feature Attribution 使用指南

## 📖 概述

本文档介绍基于梯度的 SAE 特征归因方法，这是一种高效的特征重要性分析方法，可以作为 Ablation 方法的补充或替代。

### 什么是 Gradient-Based Attribution？

Gradient-Based Attribution 通过计算模型输出对 SAE 特征激活的梯度来衡量每个特征的重要性：

- **核心思想**：如果一个特征对模型预测很重要，那么改变这个特征会显著影响输出
- **计算方式**：$\text{attribution}_i = \frac{\partial \text{output}}{\partial \text{feature}_i} \times \text{feature}_i$
- **优势**：速度快（单次前向+反向传播），可以同时分析所有特征

### 与 Ablation 方法的对比

| 特性 | Gradient-Based | Ablation |
|------|----------------|----------|
| **速度** | 快（秒级） | 慢（分钟级） |
| **性质** | 相关性（correlation） | 因果性（causation） |
| **粒度** | 连续分数 | 二元效应 |
| **可扩展性** | 可分析所有特征 | 仅分析特定特征组 |
| **证据强度** | 中等 | 强 |
| **推荐用途** | 快速筛选 | 验证假设 |

## 🚀 快速开始

### 1. 运行 Gradient Attribution

```bash
# 基本用法
bash run_gradient_attribution.sh \
  ./sae_results/L8_F2048_20260103_123456/ \
  100 \
  grad_x_act

# 参数说明
# $1: SAE checkpoint 目录
# $2: 评估样本数（100 用于快速测试，500+ 用于正式实验）
# $3: 归因方法（grad_x_act, grad_norm, integrated_gradients）
```

### 2. 查看结果

```bash
# 查看摘要
cat ./sae_results/L8_F2048_*/gradient_attribution/gradient_summary_*.txt

# 查看可视化
ls ./sae_results/L8_F2048_*/gradient_attribution/*.png
```

### 3. 与 Ablation 结果对比

```bash
python compare_methods.py \
  -g ./sae_results/L8_F2048_*/gradient_attribution/gradient_attribution_*.json \
  -a ./sae_results/L8_F2048_*/ablation_results/ablation_results_*.json
```

## 📊 输出结果

### 文件结构

```
sae_results/L8_F2048_*/
└── gradient_attribution/
    ├── gradient_attribution_grad_x_act_*.json      # 归因结果
    ├── gradient_summary_grad_x_act_*.txt           # 文本摘要
    ├── detailed_attributions_grad_x_act_*.npz      # 详细的逐样本归因
    ├── attribution_distribution_grad_x_act.png     # 归因分布
    ├── top_features_grad_x_act.png                 # Top-K 特征
    ├── spatial_comparison_grad_x_act.png           # Spatial vs Non-Spatial
    ├── attribution_heatmap_grad_x_act.png          # 归因热图
    └── feature_overlap_grad_x_act.png              # 特征重叠分析
```

### 关键指标

| 指标 | 说明 | 期望值 |
|------|------|--------|
| **Spatial/Non-Spatial Ratio** | Spatial features 的平均归因 / Non-spatial features 的平均归因 | > 1.5x |
| **Top-K Overlap** | Top-K 归因特征与已识别 spatial features 的重叠率 | > 60% |
| **Accuracy** | 模型基线准确率 | > 0.70 |

**判断标准：**
- ✅ **强一致性**：Ratio > 2.0x AND Overlap > 70%
- ~ **中等一致性**：Ratio > 1.5x OR Overlap > 50%
- ✗ **弱一致性**：Ratio < 1.5x AND Overlap < 50%

## 🔬 三种归因方法

### 1. Gradient × Activation (推荐)

**公式：** $\text{attribution}_i = \frac{\partial L}{\partial h_i} \times h_i$

**特点：**
- 最常用的归因方法
- 考虑了梯度和激活值两个因素
- 速度快，效果好

**使用场景：** 默认选择，适合大多数情况

```bash
bash run_gradient_attribution.sh ./sae_results/L8_F2048_*/ 100 grad_x_act
```

### 2. Gradient Norm

**公式：** $\text{attribution}_i = \left|\frac{\partial L}{\partial h_i}\right|$

**特点：**
- 只考虑梯度大小，不考虑激活值
- 对非稀疏特征更友好
- 可能高估不活跃特征的重要性

**使用场景：** 当特征激活很稀疏时

```bash
bash run_gradient_attribution.sh ./sae_results/L8_F2048_*/ 100 grad_norm
```

### 3. Integrated Gradients

**公式：** $\text{attribution}_i = h_i \times \int_0^1 \frac{\partial f(\alpha h)}{\partial h_i} d\alpha$

**特点：**
- 理论上更精确（满足公理化性质）
- 沿路径积分梯度，减少噪声
- 计算成本高（需要多次前向+反向传播）

**使用场景：** 需要最精确的归因，且有充足计算资源

```bash
bash run_gradient_attribution.sh ./sae_results/L8_F2048_*/ 100 integrated_gradients
```

## 📈 实验工作流

### 推荐工作流：Gradient + Ablation

```bash
# Step 1: 训练 SAE（见 README.md）
python train_sae_saelens_new.py ...

# Step 2: 分析特征，识别 spatial features（见 README.md）
python analyze_features_saelens_new.py ...

# Step 3: 用 Gradient 方法快速筛选重要特征 ⭐ NEW
bash run_gradient_attribution.sh ./sae_results/L8_F2048_*/ 500 grad_x_act

# Step 4: 用 Ablation 方法验证因果关系
bash run_ablation_new.sh ./sae_results/L8_F2048_*/ 100 simple

# Step 5: 对比两种方法的结果
python compare_methods.py \
  -g ./sae_results/L8_F2048_*/gradient_attribution/gradient_attribution_*.json \
  -a ./sae_results/L8_F2048_*/ablation_results/ablation_results_*.json
```

### 快速测试（5 分钟）

```bash
# 只用 gradient 方法（最快）
bash run_gradient_attribution.sh ./sae_results/L8_F2048_*/ 50 grad_x_act
```

**预计时间：** 2-3 分钟

### 完整验证（30 分钟）

```bash
# Gradient + Ablation
bash run_gradient_attribution.sh ./sae_results/L8_F2048_*/ 500 grad_x_act
bash run_ablation_new.sh ./sae_results/L8_F2048_*/ 500 simple
python compare_methods.py -g ... -a ...
```

**预计时间：** 15 + 15 分钟

### 最精确分析（1 小时）

```bash
# 尝试所有归因方法
for method in grad_x_act grad_norm integrated_gradients; do
  bash run_gradient_attribution.sh ./sae_results/L8_F2048_*/ 500 $method
done

# Ablation 验证
bash run_ablation_new.sh ./sae_results/L8_F2048_*/ 500 simple
```

## 💡 使用技巧

### 1. 选择合适的样本数

| 样本数 | 用途 | 预计时间 |
|--------|------|----------|
| 50 | 快速测试 | 1-2 分钟 |
| 100 | 初步分析 | 2-3 分钟 |
| 500 | 正式实验 | 5-10 分钟 |
| 1000+ | 论文发表 | 10-20 分钟 |

### 2. 选择合适的归因方法

**快速筛选：** 用 `grad_x_act`（最快，效果好）

**精确分析：** 用 `integrated_gradients`（慢但更准确）

**诊断问题：** 用 `grad_norm`（不受激活稀疏性影响）

### 3. 解读结果

**Spatial/Non-Spatial Ratio:**
- \> 2.0x：Spatial features 明显更重要 ✅
- 1.5-2.0x：Spatial features 略微更重要 ~
- < 1.5x：没有显著差异 ✗

**Top-K Overlap:**
- \> 70%：Gradient 方法与 Spatial features 高度一致 ✅
- 50-70%：中等一致性 ~
- < 50%：一致性较弱 ✗

### 4. 常见问题诊断

#### 问题 1: Spatial Ratio < 1.0（spatial features 归因反而更低）

**可能原因：**
- Spatial features 识别不准确
- 模型没有真正使用 spatial features
- 梯度噪声过大

**解决方案：**
1. 检查 `dimension_features.json` 的质量
2. 尝试 `integrated_gradients` 方法（更稳定）
3. 增加评估样本数以减少噪声

#### 问题 2: Top-K Overlap 很低（< 30%）

**可能原因：**
- Gradient 方法识别出了其他重要特征
- Spatial features 的识别阈值设置过严格/宽松

**解决方案：**
1. 查看 `top_features_*.png`，看 top features 是否合理
2. 调整 `analyze_features_saelens_new.py` 中的相关性阈值
3. 手动检查 top gradient features 的激活模式

#### 问题 3: 准确率异常低（< 0.5）

**可能原因：**
- 评估数据格式不匹配
- Option tokens 编码错误

**解决方案：**
1. 检查 `OPTION_TOKENS` 的输出
2. 确认评估数据的 `correct_option` 字段正确
3. 尝试用少量样本调试

## 🔬 原理详解

### Gradient × Activation 的直觉

考虑一个简单的线性模型：$y = \sum_i w_i x_i$

- **梯度** $\frac{\partial y}{\partial x_i} = w_i$ 表示：改变 $x_i$ 对输出的影响率
- **激活** $x_i$ 表示：当前特征的激活强度
- **归因** $w_i \times x_i$ 表示：特征 $i$ 对当前输出的实际贡献

对于神经网络，虽然不是线性的，但这个直觉仍然成立：
- 高梯度 = 敏感位置
- 高激活 = 强信号
- 高归因 = 重要特征

### 为什么 Gradient ≠ Ablation？

**Gradient 方法：**
- 测量局部导数（在当前激活点的切线斜率）
- 假设小扰动，线性近似
- 快但可能不准确（非线性效应）

**Ablation 方法：**
- 测量全局效应（完全移除特征）
- 不做线性假设
- 慢但更可靠（真实因果效应）

**类比：**
- Gradient = "如果我稍微移动一下，会怎样？"
- Ablation = "如果我完全拿走这个，会怎样？"

### Integrated Gradients 的改进

普通梯度在某些情况下会失效（如饱和区域）。Integrated Gradients 通过沿路径积分来解决：

$$\text{IG}_i = (x_i - x_i^{\text{baseline}}) \times \int_0^1 \frac{\partial f(x^{\text{baseline}} + \alpha (x - x^{\text{baseline}}))}{\partial x_i} d\alpha$$

**优点：**
- 满足公理化性质（Completeness, Sensitivity）
- 对饱和区域更鲁棒
- 减少梯度噪声

**缺点：**
- 需要多次前向+反向传播（慢 50x）
- 对 baseline 的选择敏感

## 📚 参考文献

1. **Gradient × Input**
   - Shrikumar et al., "Learning Important Features Through Propagating Activation Differences" (2017)

2. **Integrated Gradients**
   - Sundararajan et al., "Axiomatic Attribution for Deep Networks" (2017)
   - Paper: https://arxiv.org/abs/1703.01365

3. **SAE Attribution**
   - Anthropic, "Towards Monosemanticity" (2023)
   - Discusses feature attribution in SAEs

## 🎯 总结

### 何时使用 Gradient 方法？

✅ **推荐使用：**
- 需要快速筛选大量特征
- 想了解所有特征的相对重要性
- 需要连续的重要性分数（而非二元）
- 计算资源有限

❌ **不推荐使用：**
- 需要强因果证据（用 Ablation）
- 特征效应高度非线性
- 模型在饱和区域（考虑 Integrated Gradients）

### 最佳实践

1. **初步筛选：** 用 Gradient (`grad_x_act`) 快速识别重要特征
2. **因果验证：** 用 Ablation 验证 top-K 特征的因果作用
3. **综合分析：** 结合两种方法的结果，互相验证

### 方法组合建议

| 目标 | 推荐组合 | 预计时间 |
|------|----------|----------|
| **快速探索** | Gradient only | 5 分钟 |
| **标准验证** | Gradient + Ablation (simple) | 25 分钟 |
| **论文级别** | Gradient + Ablation + Compare | 45 分钟 |
| **最高精度** | Integrated Gradients + Ablation (full) | 2 小时 |

---

**下一步：** 尝试运行 `bash run_gradient_attribution.sh` 来分析你的 SAE 特征！




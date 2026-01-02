# SAE Feature Ablation 实验指南

## 概述

本指南介绍如何使用 SAE feature ablation 实验来验证 spatial features 的因果作用。

## 实验原理

### 核心思路

```
                        Forward Pass
Model Input → ... → Layer 8 → ... → Output
                       ↓
                    SAE Hook
                       ↓
              1. Encode: x → h
              2. Ablate: h[spatial_ids] = 0
              3. Decode: h → x'
              4. Replace: x ← x'
```

### 三种实验条件

1. **No Ablation (Baseline)**: 不干预，测量模型原始性能
2. **Spatial Ablation**: 消除 spatial features，测量性能下降
3. **Random Ablation (Control)**: 消除随机 features（同等数量），作为对照

### 预期结果

如果 spatial features 真的对空间推理有因果作用，我们应该看到：

```
Acc(No Ablation) > Acc(Random Ablation) >> Acc(Spatial Ablation)
```

即：**消除 spatial features 比消除随机 features 导致更大的性能下降**。

## 前置条件

在运行 ablation 实验之前，你需要：

1. ✅ 训练好 SAE（`train_sae_saelens.py`）
2. ✅ 分析并识别 spatial features（`analyze_features_saelens.py`）

确保以下文件存在：
```
sae_results/
└── L8_F2048_L10.0001_20260102_120000/
    ├── sae_checkpoint.pt          # SAE 权重
    ├── config.json                 # 训练配置
    └── analysis/
        ├── dimension_features.json # Spatial feature IDs
        └── feature_stats.npz       # Feature 统计
```

## 使用方法

### 方法 1：使用脚本（推荐）

最简单的方式：

```bash
bash run_ablation.sh <sae_checkpoint_dir> [max_samples] [version]
```

**参数说明：**
- `sae_checkpoint_dir`: SAE checkpoint 目录路径
- `max_samples`: 评估样本数（可选，默认 100）
- `version`: 实验版本（可选，默认 `simple`）

**示例：**

```bash
# 使用简化版本，评估 100 个样本
bash run_ablation.sh ./sae_results/L8_F2048_L10.0001_20260102_120000 100 simple

# 全量评估
bash run_ablation.sh ./sae_results/L8_F2048_L10.0001_20260102_120000 none simple

# 使用完整版本（基于生成）
bash run_ablation.sh ./sae_results/L8_F2048_L10.0001_20260102_120000 100 full
```

### 方法 2：直接运行 Python 脚本

#### 简化版本（推荐）

更快、更稳定，基于 logits 直接预测：

```bash
python ablate_features_simple.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --sae_checkpoint ./sae_results/L8_F2048_L10.0001_20260102_120000 \
    --eval_data_file ../../data/spatial/test_data.json \
    --max_samples 100
```

#### 完整版本

基于文本生成，更接近真实使用场景：

```bash
python ablate_features_saelens.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --sae_checkpoint ./sae_results/L8_F2048_L10.0001_20260102_120000 \
    --eval_data_file ../../data/spatial/test_data.json \
    --max_samples 100
```

## 两个版本的区别

### Simple Version（推荐）

- **优点：**
  - ✅ 更快（直接用 logits）
  - ✅ 更稳定（不依赖生成）
  - ✅ 更可控（确定性预测）
  
- **缺点：**
  - ❌ 不反映真实生成场景
  - ❌ 只能用于分类任务

- **适用场景：**
  - 快速验证 spatial features 的因果作用
  - 大规模实验（需要评估很多样本）
  - 资源受限环境

### Full Version

- **优点：**
  - ✅ 更接近真实使用（生成文本）
  - ✅ 可以看到实际输出
  - ✅ 更有说服力

- **缺点：**
  - ❌ 更慢（需要生成多个 tokens）
  - ❌ 可能不稳定（生成可能失败）
  - ❌ 内存占用更大

- **适用场景：**
  - 最终结果验证
  - 需要展示实际输出
  - 有充足计算资源

## 输出结果

### 文件结构

```
sae_results/L8_F2048_L10.0001_20260102_120000/
└── ablation_results/
    ├── ablation_summary_20260102_150000.txt   # 简要结果
    └── ablation_results_20260102_150000.json  # 详细结果
```

### Summary 文件示例

```
============================================================
SAE Feature Ablation Results
============================================================

Model: Qwen/Qwen2.5-7B-Instruct
Layer: 8
Spatial features: 45
Eval samples: 100

============================================================
Accuracy Results:
============================================================
Baseline           : 0.8500
Spatial ablation   : 0.6200 (-0.2300)
Random ablation    : 0.8100 (-0.0400)

Relative drops:
  Spatial: 27.06%
  Random:  4.71%

============================================================
Conclusion:
============================================================
✓ Spatial features are causally important.
```

### JSON 结果文件

包含详细信息：
- 配置参数
- Spatial feature IDs
- 三种条件的准确率
- 示例输出（前 10 个样本）
- 答案 logits 分布

## 结果解读

### 判断标准

**强因果证据：**
```
Spatial drop >> Random drop
例如：Spatial: -27%, Random: -5%
```
→ **Spatial features 对空间推理是必要的**

**弱因果证据：**
```
Spatial drop ≈ Random drop
例如：Spatial: -10%, Random: -8%
```
→ **Spatial features 重要，但不是唯一重要的**

**无因果证据：**
```
Spatial drop ≤ Random drop 或 ≈ 0
例如：Spatial: -2%, Random: -3%
```
→ **Spatial features 可能只是相关，不是因果**

### 统计显著性

对于小样本（N < 100），需要考虑统计波动：
- **建议：** 运行多次实验（不同随机种子）
- **计算：** 置信区间或 p-value

对于大样本（N > 500），结果更可靠。

## 高级用法

### 1. 分维度 Ablation

可以单独消除某个空间维度的特征，例如只消除 "left/right" features：

```python
# 修改 ablate_features_simple.py
# 替换 spatial_feature_ids 为特定维度的 IDs
x_features = [f['feature_idx'] for f in dimension_features['X_positive']]
x_features += [f['feature_idx'] for f in dimension_features['X_negative']]
```

### 2. 渐进式 Ablation

逐步增加消除的特征数量，绘制 accuracy vs. #ablated features 曲线：

```python
for k in [5, 10, 20, 50, 100, 200]:
    top_k_features = spatial_feature_ids[:k]
    ablator = SAEAblator(W_enc, b_enc, W_dec, b_dec, top_k_features)
    results = evaluate_with_ablation(model, eval_data, ablator)
    print(f"Ablate top {k}: Acc = {results['accuracy']:.4f}")
```

### 3. 特征注入（Activation Addition）

除了消除特征，还可以增强特征：

```python
# 在 SAEAblator.__call__ 中
h[:, self.feature_ids] *= 2.0  # 增强 2 倍
# 或
h[:, self.feature_ids] += 1.0  # 添加常数激活
```

## 常见问题

### Q1: Baseline accuracy 太低（< 50%）？

**可能原因：**
- 评估数据质量问题
- 模型没有学到空间推理能力
- ANSWER_TOKENS 映射不正确

**解决方案：**
- 检查 `ANSWER_TOKENS` 中的 token IDs 是否正确
- 先在原始模型上测试（不加 hook）
- 检查评估数据格式

### Q2: Spatial ablation 和 Random ablation 效果一样？

**可能原因：**
- Spatial features 识别不准确
- 特征数量太少（< 10）
- 模型对这些特征不敏感

**解决方案：**
- 降低 `CORR_THRESHOLD`，识别更多 spatial features
- 检查 `dimension_features.json` 中的特征质量
- 尝试不同的 layer

### Q3: 内存不足？

**解决方案：**
- 使用更小的 `--max_samples`
- 使用 `simple` 版本（内存占用更小）
- 降低 batch size（需要修改代码）

### Q4: 运行太慢？

**解决方案：**
- 使用 `simple` 版本（快 5-10x）
- 减少 `--max_samples`
- 使用更快的 GPU

## 下一步

完成 ablation 实验后，可以：

1. **撰写论文：** 将结果整理成论文/报告
2. **可视化：** 绘制 ablation 效果图（accuracy bar chart）
3. **深入分析：** 
   - 哪些样本最受影响？
   - 不同空间关系（left/right vs. above/below）的差异？
   - Feature activation patterns 分析
4. **扩展实验：**
   - 尝试其他 layers
   - 尝试其他模型
   - 尝试其他任务（例如数学推理）

## 引用

如果你使用了这些代码，请引用相关工作：
- SAELens: [https://github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)
- TransformerLens: [https://github.com/neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens)

## 联系

如有问题或建议，请联系项目维护者。


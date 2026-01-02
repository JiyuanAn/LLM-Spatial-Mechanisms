# 🎉 SAE 训练完成 - 后续分析指南

恭喜！您已经成功训练了一个 Sparse Autoencoder (SAE)。现在可以开始分析模型学到的特征。

## 📁 训练结果

**输出目录**：`./sae_experiment_20260102_005950/`

**包含文件**：
- `sae_step_*.pt` - 各个训练步骤的检查点（500, 1000, ..., 5000）
- `train_config.json` - 训练配置
- 最终模型：`sae_step_5000.pt` ✨

**训练效果**：
- ✅ MSE: 0.0006 → 0.0004（重建误差下降）
- ✅ 稀疏度: 0.045 → 0.026（特征激活更稀疏）
- ✅ 训练稳定，损失持续下降

---

## 🔍 Step 1: 分析所有特征统计

首先了解哪些特征最重要、最活跃：

```bash
cd /home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae

python analyze_features.py \
    --checkpoint ./sae_experiment_20260102_005950/sae_step_5000.pt \
    --model_path "/home/s202507009/workspace/LLMs/Qwen2.5/Qwen2.5-7B-Instruct" \
    --data_path "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/dataGenerate/spatial_reasoning_dataset_EN.json" \
    --analyze_all \
    --output feature_statistics.json \
    --batch_size 8
```

**输出内容**：
- Top 20 最频繁激活的特征
- Top 20 平均激活值最高的特征
- 死特征（从不激活）数量
- 完整统计保存到 `feature_statistics.json`

**预计时间**：2-5 分钟

---

## 📊 Step 2: 可视化特征统计

查看特征的整体分布和特性：

```bash
python visualize_features.py \
    --stats feature_statistics.json \
    --top_n 30
```

**输出内容**：
- 最常激活的特征列表
- 最强激活的特征列表
- 稀疏度统计
- 死特征分析

---

## 🔎 Step 3: 分析特定特征

根据统计结果，选择感兴趣的特征进行深入分析：

```bash
# 分析特征 42（替换为您感兴趣的特征 ID）
python analyze_features.py \
    --checkpoint ./sae_experiment_20260102_005950/sae_step_5000.pt \
    --model_path "/home/s202507009/workspace/LLMs/Qwen2.5/Qwen2.5-7B-Instruct" \
    --data_path "/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/dataGenerate/spatial_reasoning_dataset_EN.json" \
    --feature_id 42 \
    --top_k 20 \
    --output feature_42_examples.json
```

**输出内容**：
- Top 20 激活该特征最强的输入样本
- 可以帮助理解该特征代表什么语义

---

## 📝 Step 4: 可视化和解释特征

详细分析特征激活的模式：

```bash
python visualize_features.py \
    --feature feature_42_examples.json \
    --output feature_42_report.txt
```

**输出内容**：
- 该特征的统计信息
- 常见的空间关系词（left, right, above, below 等）
- Top 样本的详细展示
- 保存详细报告到文本文件

---

## 🚀 Step 5: 批量分析多个特征

一次性分析多个感兴趣的特征：

```bash
bash batch_analyze.sh
```

这会：
1. 生成所有特征的统计
2. 分析多个具体特征（特征 0, 1, 2, 3, 4, 5, 10, 20, 50, 100）
3. 将所有结果保存到 `feature_analysis_results/` 目录

**预计时间**：10-20 分钟

---

## 🎯 分析建议

### 寻找空间推理相关的特征

您的数据集是关于空间推理的，可以重点关注：

1. **方向特征**
   - 寻找对 "left/right" 敏感的特征
   - 寻找对 "above/below" 敏感的特征
   - 寻找对 "front/behind" 敏感的特征

2. **复杂推理特征**
   - 寻找对多步推理敏感的特征（长句子）
   - 寻找对实体数量敏感的特征

3. **答案类型特征**
   - 寻找与答案相关的特征

### 分析流程建议

```bash
# 1. 先看整体统计
python analyze_features.py --checkpoint ... --analyze_all --output stats.json
python visualize_features.py --stats stats.json --top_n 50

# 2. 从统计中选择 top 10 最频繁的特征
# 假设是: 10, 25, 42, 78, 103, 156, 234, 567, 890, 1234

# 3. 逐个分析这些特征
for FID in 10 25 42 78 103 156 234 567 890 1234; do
    python analyze_features.py --checkpoint ... --feature_id $FID --output feature_${FID}.json
    python visualize_features.py --feature feature_${FID}.json --output feature_${FID}_report.txt
done

# 4. 手动查看报告，识别有意义的特征
ls -lh feature_*_report.txt
cat feature_10_report.txt
cat feature_25_report.txt
# ...
```

---

## 📈 进阶分析

### 1. 比较不同层的 SAE

训练不同层（如 layer 4, 8, 12, 16, 20）的 SAE：

```bash
# 修改 example_train.sh 中的 LAYER 参数
LAYER=4 bash example_train.sh
LAYER=12 bash example_train.sh
LAYER=16 bash example_train.sh
```

然后比较不同层学到的特征有何不同。

### 2. 调整稀疏度

尝试不同的 L1 系数：

```bash
# 更稀疏（更少特征激活）
L1_COEF=5e-4 bash example_train.sh

# 更密集（更多特征激活）
L1_COEF=1e-4 bash example_train.sh
```

### 3. 增加特征数量

训练更大的 SAE：

```bash
# 8192 特征
N_FEATURES=8192 bash example_train.sh

# 16384 特征
N_FEATURES=16384 bash example_train.sh
```

### 4. 与探测器（Probe）结合

将 SAE 特征与您的空间推理探测器结合：
- 使用 SAE 特征作为探测器的输入
- 看看哪些 SAE 特征对空间推理最有帮助

---

## 🛠️ 实用命令速查

```bash
# 快速查看训练配置
cat sae_experiment_20260102_005950/train_config.json

# 加载检查点
python
>>> import torch
>>> ckpt = torch.load("sae_experiment_20260102_005950/sae_step_5000.pt")
>>> ckpt.keys()
>>> ckpt["cfg"]

# 清理旧实验（可选）
rm -rf sae_experiment_20260102_005558
rm -rf sae_experiment_20260102_005749

# 创建结果目录
mkdir -p results/layer8_4096features
mv sae_experiment_20260102_005950 results/layer8_4096features/
mv feature_*.json results/layer8_4096features/
mv feature_*_report.txt results/layer8_4096features/
```

---

## 📚 相关文档

- `README.md` - 基本使用说明
- `train_sae.py` - 训练脚本
- `analyze_features.py` - 分析脚本
- `visualize_features.py` - 可视化脚本

---

## ❓ 常见问题

**Q: 如何判断特征质量好坏？**
A: 看这几个指标：
- 死特征比例（< 20% 较好）
- 重建误差（MSE < 0.001 较好）
- 特征可解释性（top 样本有明显共同模式）

**Q: 需要分析所有 4096 个特征吗？**
A: 不需要。通常分析 top 20-50 个最活跃的特征就能获得很多洞察。

**Q: 如何使用训练好的 SAE？**
A: 可以：
1. 分析模型内部表示
2. 作为特征提取器用于下游任务
3. 用于模型可解释性研究
4. 与探测器（probe）结合分析特定能力

---

## 🎓 下一步学习

推荐阅读：
- Anthropic 的 SAE 论文："Towards Monosemanticity"
- OpenAI 的可解释性研究
- TransformerLens 文档

祝分析顺利！🚀


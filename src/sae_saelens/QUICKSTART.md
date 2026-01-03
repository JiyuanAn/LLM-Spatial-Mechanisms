# SAELens 快速开始指南 ⚡

## 5 分钟快速上手

### 步骤 1: 安装 SAELens

```bash
cd /home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae_saelens

# 方法1: 从 PyPI 安装 (推荐)
pip install sae-lens

# 方法2: 从 requirements.txt 安装
pip install -r requirements.txt

# 验证安装
python -c "import sae_lens; print(f'SAELens {sae_lens.__version__} installed')"
```

### 步骤 2: 运行实验

```bash
# 一键运行（最简单）
chmod +x run_experiment.sh
bash run_experiment.sh
```

**就这么简单！** 🎉

---

## 手动运行（如果需要）

### 训练 SAE

```bash
python train_sae_saelens.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr "../dataGenerate/spatial_reasoning_dataset_ZH.json" \
  -te "../dataGenerate/spatial_reasoning_dataset_ZH_test.json"
```

### 分析特征

```bash
# 找到最新的 checkpoint
LATEST=$(ls -td sae_results/L8_* | head -1)

# 运行分析
python analyze_features_saelens.py -c "$LATEST"
```

---

## 预期输出

### 训练阶段

```
Epoch 1/5
  Train - Loss: 0.0816, MSE: 0.0626, L1: 0.0190, L0: 467.6
  Test  - Loss: 0.0356, MSE: 0.0167, L1: 0.0189, L0: 349.2

Epoch 5/5
  Train - Loss: 0.0158, MSE: 0.0059, L1: 0.0099, L0: 217.3
  Test  - Loss: 0.0148, MSE: 0.0053, L1: 0.0095, L0: 205.7
```

### 分析阶段

```
Spatial features found:
  Right     :  28 features
  Left      :  36 features
  Above     :  24 features
  Below     :  30 features
  Front     :   2 features
  Behind    :   6 features

Sparse spatial features: 17
```

---

## 成功标准 ✅

| 指标 | 期望值 | 说明 |
|------|--------|------|
| L0 | 50-250 | 不崩溃到 0 或爆炸到 1000+ |
| 稀疏特征 | >15 | 激活率 1-50% 的空间特征 |
| 空间方向 | 5-6/6 | 覆盖的方向数 |
| 相关性 | 0.15-0.25 | 特征与维度的相关系数 |

---

## 对比：SAELens vs 自实现

### SAELens 版本（本目录）⭐

```bash
cd sae_saelens
bash run_experiment.sh
```

**优点**:
- ✅ 官方支持，稳定可靠
- ✅ 标准化配置
- ✅ 适合正式实验

### 自实现版本

```bash
cd ../sae
bash run_experiment.sh
```

**优点**:
- ✅ 完全控制实现细节
- ✅ 易于理解和定制
- ✅ 适合学习研究

**推荐**: 论文用 SAELens，学习用自实现

---

## 常见问题速查 🔧

### 问题 1: ImportError: No module named 'sae_lens'

```bash
pip install sae-lens
# 或
pip install git+https://github.com/jbloomAus/SAELens.git
```

### 问题 2: CUDA out of memory

```bash
# 减少特征数量
python train_sae_saelens.py ... --n_features 1024

# 或使用更少样本
python train_sae_saelens.py ... --max_samples 5000
```

### 问题 3: L0 崩溃到 0

```bash
# 降低 L1 系数
python train_sae_saelens.py ... --l1_coeff 5e-5
```

### 问题 4: 训练太慢

```bash
# 减少训练 tokens
python train_sae_saelens.py ... --num_tokens 50000

# 或限制样本数
python train_sae_saelens.py ... --max_samples 3000
```

---

## 下一步 🚀

实验成功后，可以：

1. **可视化特征**
   ```bash
   open sae_results/L8_*/analysis/*.png
   ```

2. **检查稀疏特征**
   ```bash
   cat sae_results/L8_*/analysis/dimension_features.json | jq
   ```

3. **因果干预**（进阶）
   - 消融特定 features
   - 测试因果作用

4. **跨模型验证**
   - 在其他模型上重复
   - 验证发现的普遍性

---

## 时间估算 ⏱️

| 步骤 | 时间 | 说明 |
|------|------|------|
| 安装 | 2-5 分钟 | 取决于网速 |
| 训练 | 15-30 分钟 | 取决于数据量和 GPU |
| 分析 | 5-10 分钟 | |
| **总计** | **~30 分钟** | |

---

## 快速命令参考 📝

```bash
# 完整流程
bash run_experiment.sh

# 只训练
python train_sae_saelens.py -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr ../dataGenerate/spatial_reasoning_dataset_ZH.json \
  -te ../dataGenerate/spatial_reasoning_dataset_ZH_test.json

# 只分析
python analyze_features_saelens.py -c sae_results/L8_F2048_*/

# 查看结果
cat sae_results/L8_F2048_*/training_history.json
cat sae_results/L8_F2048_*/analysis/dimension_features.json

# 查看可视化
ls sae_results/L8_F2048_*/analysis/*.png
```

---

## 需要帮助？

查看完整文档：`README.md`

**开始实验吧！** 🎊




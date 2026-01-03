# SAE 实验快速开始指南

## 5 分钟快速开始 ⚡

### 1️⃣ 确认环境

```bash
cd /home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae

# 确认数据文件存在
ls ../dataGenerate/spatial_reasoning_dataset_ZH*.json

# 安装依赖（如果还没安装）
pip install -r requirements.txt
```

### 2️⃣ 运行完整实验（一键）

```bash
# 使用默认配置运行
bash run_experiment.sh

# 或者手动分步运行（见下方）
```

这将自动完成：
- ✅ 训练 SAE（Layer 8, 2048 features）
- ✅ 分析特征
- ✅ 生成可视化

**预计时间**: 15-30 分钟

### 3️⃣ 查看结果

```bash
# 训练曲线
open sae_results/L8_F2048_*/training_curves.png

# Top features
cat sae_results/L8_F2048_*/analysis/top_features.json | head -50

# 空间特征分类
cat sae_results/L8_F2048_*/analysis/dimension_features.json

# Probe 性能
cat sae_results/L8_F2048_*/analysis/probe_results.json
```

---

## 分步执行指南 📋

### Step 1: 训练 SAE

```bash
python train_sae.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr "../dataGenerate/spatial_reasoning_dataset_ZH.json" \
  -te "../dataGenerate/spatial_reasoning_dataset_ZH_test.json" \
  --layer 8 \
  --n_features 2048 \
  --l1_coeff 1e-3 \
  --num_epochs 5
```

**关键参数**:
- `--layer`: 目标层（默认 8）
- `--n_features`: SAE 特征数（默认 2048）
- `--l1_coeff`: L1 稀疏度系数（默认 1e-3）
- `--num_epochs`: 训练轮数（默认 5）
- `--max_samples`: 限制训练样本数（用于快速测试）

**输出**: `sae_results/L8_F2048_L1*_*/`

### Step 2: 分析特征

```bash
# 替换 <checkpoint_path> 为实际路径
python analyze_features.py \
  -c "sae_results/L8_F2048_L1*_*/sae_checkpoint.pt" \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -te "../dataGenerate/spatial_reasoning_dataset_ZH_test.json" \
  --top_k 50
```

**输出**: `sae_results/L8_F2048_L1*_*/analysis/`

### Step 3: 推理测试（可选）

```bash
# 单个 prompt
python inference_example.py \
  -c "sae_results/L8_F2048_L1*_*/sae_checkpoint.pt" \
  -p "甲在乙的左边。乙在丙的上面。甲在丙的什么位置？"

# 批量测试
python inference_example.py \
  -c "sae_results/L8_F2048_L1*_*/sae_checkpoint.pt" \
  -pf "../dataGenerate/spatial_reasoning_dataset_ZH_test.json"
```

---

## 超参数搜索 🔍

如果初始结果不理想，运行网格搜索：

```bash
bash grid_search.sh
```

这将尝试不同的 `l1_coeff` 和 `n_features` 组合，并给出最佳配置建议。

**预计时间**: 2-3 小时（9 个实验）

### 比较多个实验

```bash
python compare_experiments.py \
  -e sae_results_grid/L8_F2048_* \
  -o comparison.png
```

---

## 结果评估标准 ✅

### 训练成功的标志

| 指标 | 目标范围 | 含义 |
|-----|---------|------|
| **L0** | 20-100 | 平均激活特征数（稀疏度） |
| **Cosine Similarity** | >0.95 | 重构质量 |
| **Probe R²** | >0.20 | 特征预测能力 |
| **空间特征数** | >30 | 找到的方向相关特征 |

### 如何判断实验成功？

✅ **成功**:
```
L0: 45.2
Cosine Similarity: 0.968
Probe R² (all): 0.254
Probe R² (top 50): 0.241
Spatial features: Left(8), Right(7), Above(6), Below(9), Front(5), Behind(6)
```

❌ **需要调整**:
```
L0: 320  ← 太高，增加 l1_coeff
CosSim: 0.87  ← 重构差，减少 l1_coeff
Probe R²: 0.12  ← 预测差，增加 features 或换 layer
Spatial features: 5  ← 太少，检查数据和 hook 点
```

---

## 常见问题解决 🔧

### Q1: GPU 内存不足

```bash
# 解决方案 1: 减小 batch size
python train_sae.py ... --batch_size 64

# 解决方案 2: 减少 features
python train_sae.py ... --n_features 1024

# 解决方案 3: 限制训练样本
python train_sae.py ... --max_samples 5000
```

### Q2: L0 太高（不够稀疏）

```bash
# 增加 L1 系数
python train_sae.py ... --l1_coeff 3e-3  # 或 5e-3
```

### Q3: 重构质量差（CosSim < 0.9）

```bash
# 减少 L1 系数
python train_sae.py ... --l1_coeff 5e-4

# 或增加特征数
python train_sae.py ... --n_features 4096
```

### Q4: 找不到空间相关特征

1. **检查 probe 基线**: 确保 layer sweep probe 在 Layer 8 有峰值 (R²~0.27)
2. **检查数据**: 确认数据文件正确且有 target 字段
3. **尝试其他 hook 点**: 将 `hook_mlp_out` 改为 `hook_resid_post`
4. **增加训练时间**: `--num_epochs 10`

### Q5: 训练太慢

```bash
# 快速测试模式
python train_sae.py ... \
  --max_samples 3000 \
  --num_epochs 3 \
  --batch_size 256
```

---

## 目录结构

```
src/sae/
├── train_sae.py              # 主训练脚本
├── analyze_features.py       # 特征分析
├── inference_example.py      # 推理示例
├── compare_experiments.py    # 实验对比
├── run_experiment.sh         # 一键运行
├── grid_search.sh            # 超参数搜索
├── requirements.txt          # 依赖
├── README.md                 # 详细文档
├── QUICKSTART.md            # 本文档
└── sae_results/             # 结果输出（自动创建）
    └── L8_F2048_L1*_*/
        ├── sae_checkpoint.pt       # SAE 权重
        ├── config.json             # 配置
        ├── training_history.json   # 训练历史
        ├── training_curves.png     # 训练曲线
        └── analysis/               # 分析结果
            ├── top_features.json
            ├── dimension_features.json
            ├── probe_results.json
            └── *.png
```

---

## 下一步建议 🚀

### 实验完成后，你可以：

1. **可视化 top features**
   - 看看哪些 features 对应哪些空间方向
   - 绘制 feature activation 热图

2. **因果干预实验**
   - 消融特定 features，观察模型行为变化
   - 验证 features 的因果作用

3. **跨数据集验证**
   - 在英文数据上测试
   - 在不同难度的任务上测试

4. **扩展到其他层**
   - 对比 Layer 7, 8, 9 的 features
   - 绘制层间 feature 变化

5. **Feature steering**
   - 手动激活特定 features
   - 引导模型生成特定空间推理

---

## 快速参考卡 📝

```bash
# 最简单运行
bash run_experiment.sh

# 快速测试（5分钟）
python train_sae.py -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr ../dataGenerate/spatial_reasoning_dataset_ZH.json \
  -te ../dataGenerate/spatial_reasoning_dataset_ZH_test.json \
  --max_samples 2000 --num_epochs 3

# 查看最新结果
ls -lt sae_results/ | head -5

# 快速分析
python analyze_features.py \
  -c $(ls -t sae_results/*/sae_checkpoint.pt | head -1) \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -te ../dataGenerate/spatial_reasoning_dataset_ZH_test.json
```

---

**祝实验顺利！** 🎉

如有问题，请查看完整文档 `README.md`




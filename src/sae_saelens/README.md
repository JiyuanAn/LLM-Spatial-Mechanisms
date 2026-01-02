# SAE 实验（基于 SAELens 官方库）

## 📦 概述

本目录包含基于 **SAELens** 官方库的 SAE 训练和分析代码，用于从 Qwen2.5-7B-Instruct Layer 8 中提取稀疏的空间推理特征。

### 与自实现版本的区别

| 特性 | 自实现版本 (`../sae/`) | SAELens 版本 (本目录) |
|------|----------------------|---------------------|
| **实现方式** | 手动实现 SAE 架构 | 使用 SAELens 官方库 |
| **稳定性** | 依赖自己的实现 | 官方支持，更稳定 |
| **功能** | 基本 SAE 功能 | 完整的 SAELens 生态 |
| **适用场景** | 学习和定制 | 生产和标准化实验 |
| **推荐** | 研究探索 | **正式实验** ⭐ |

## 🎯 实验目标

验证 **Layer 8** 是否包含：
1. ✅ 稀疏、可分离的空间特征
2. ✅ 方向/位置选择性 features
3. ✅ 状态保持型 features

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae_saelens

# 安装 SAELens 和依赖
pip install -r requirements.txt

# 或者单独安装 SAELens
pip install sae-lens
```

### 2. 一键运行实验

```bash
# 给脚本执行权限
chmod +x run_experiment.sh

# 运行完整流程
bash run_experiment.sh
```

**预计时间**: 20-30 分钟

### 3. 分步执行

#### Step 1: 训练 SAE

```bash
python train_sae_saelens.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr "../dataGenerate/spatial_reasoning_dataset_ZH.json" \
  -te "../dataGenerate/spatial_reasoning_dataset_ZH_test.json" \
  --layer 8 \
  --n_features 2048 \
  --l1_coeff 1e-4 \
  --num_tokens 100000
```

**参数说明**:
- `--layer`: 目标层（默认 8）
- `--n_features`: SAE 特征数量（默认 2048）
- `--l1_coeff`: L1 稀疏度系数（默认 1e-4）
- `--num_tokens`: 训练 token 数（默认 100000）
- `--max_samples`: 限制样本数（用于快速测试）

#### Step 2: 分析特征

```bash
python analyze_features_saelens.py \
  -c "sae_results/L8_F2048_*/" \
  --top_k 50
```

#### Step 3: 特征干预实验（Ablation）⭐

验证 spatial features 的因果作用：

```bash
# 使用简化版本（推荐，更快更稳定）
bash run_ablation.sh ./sae_results/L8_F2048_*/ 100 simple

# 或直接运行 Python 脚本
python ablate_features_simple.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c "./sae_results/L8_F2048_*/" \
  -e "../dataGenerate/spatial_reasoning_dataset_ZH_test.json" \
  --max_samples 100
```

**参数说明**:
- `max_samples`: 评估样本数（100 适合快速测试，500+ 用于正式实验）
- `version`: `simple`（基于 logits，快）或 `full`（基于生成，慢）

#### Step 4: 可视化结果

```bash
python visualize_ablation.py \
  -r "./sae_results/L8_F2048_*/ablation_results/ablation_results_*.json"
```

## 📊 输出结果

### 文件结构

```
sae_results/
└── L8_F2048_L10.0001_YYYYMMDD_HHMMSS/
    ├── sae_checkpoint.pt        # SAE 模型权重
    ├── activations.pt           # 保存的激活和 targets
    ├── training_history.json    # 训练历史
    ├── config.json             # 实验配置
    ├── analysis/
    │   ├── feature_stats.npz              # 特征统计
    │   ├── feature_target_correlations.npz # 相关性矩阵
    │   ├── dimension_features.json        # 空间特征分类 ⭐
    │   ├── probe_results.json             # Probe 性能
    │   ├── activation_frequency.png       # 激活频率分布
    │   └── feature_correlations.png       # 相关性热图
    └── ablation_results/                  # 干预实验结果 ⭐
        ├── ablation_summary_*.txt         # 简要总结
        ├── ablation_results_*.json        # 详细结果
        ├── ablation_accuracy_bar.png      # 准确率对比图
        ├── ablation_drop.png              # 性能下降图
        ├── ablation_combined.png          # 综合对比图
        ├── ablation_flips.png             # 样本翻转分析
        └── ablation_table.tex             # LaTeX 表格代码
```

### 关键指标

#### 训练阶段（train_sae_saelens.py）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **L0** | 20-200 | 平均激活特征数 |
| **MSE Loss** | <0.01 | 重构误差 |
| **Dead Features** | <30% | 从不激活的特征比例 |

#### 分析阶段（analyze_features_saelens.py）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **Probe R²** | >0.10 | 特征预测能力 |
| **稀疏空间特征** | >15 | 激活率 1-50% 的空间特征 |
| **覆盖维度** | 5-6/6 | 覆盖的空间方向数 |

#### 干预阶段（ablate_features_*.py）⭐ 最关键

| 指标 | 期望结果 | 说明 |
|------|---------|------|
| **Baseline Acc** | >0.70 | 原始模型性能 |
| **Spatial Drop** | >10% | 消除 spatial features 后的性能下降 |
| **Random Drop** | <5% | 消除随机 features 后的性能下降 |
| **Effect Size** | Spatial > 2×Random | 因果证据强度 |

**判断标准：**
- ✅ **强因果**：Spatial Drop > 2 × Random Drop
- ~ **弱因果**：Spatial Drop > Random Drop
- ✗ **无因果**：Spatial Drop ≤ Random Drop

## 🔍 SAELens 特性

### 优势

1. **官方实现**
   - 经过大规模验证
   - 持续维护和更新
   - 社区支持

2. **完整功能**
   - 多种 SAE 架构（standard, gated, etc.）
   - 内置训练循环
   - WandB 集成
   - 自动 checkpointing

3. **标准化**
   - 统一的配置格式
   - 可重现的实验
   - 与其他研究兼容

### 配置选项

SAELens 支持的主要配置：

```python
LanguageModelSAERunnerConfig(
    # Model
    model_name="Qwen/Qwen2.5-7B-Instruct",
    hook_point="blocks.8.hook_mlp_out",
    d_in=3584,  # 自动检测
    
    # SAE Architecture
    architecture="standard",  # or "gated", "topk"
    d_sae=2048,
    expansion_factor=None,  # 或用 d_sae
    
    # Training
    l1_coefficient=1e-4,
    lr=3e-4,
    train_batch_size_tokens=4096,
    training_tokens=100000,
    
    # Logging
    log_to_wandb=False,  # 可选：启用 WandB
)
```

## 🎓 使用建议

### 何时使用 SAELens？

✅ **推荐使用**:
- 正式实验和发表论文
- 需要标准化流程
- 需要与其他研究对比
- 追求稳定性和可重现性

❌ **不推荐**:
- 需要高度定制 SAE 架构
- 实验环境不支持 SAELens
- 想深入理解 SAE 实现细节（用自实现版本）

### 与自实现版本对比

| 场景 | 推荐版本 |
|------|---------|
| **论文实验** | SAELens ⭐ |
| **学习研究** | 自实现 |
| **快速原型** | 自实现 |
| **生产部署** | SAELens ⭐ |
| **定制架构** | 自实现 |

## 🐛 常见问题

### Q1: SAELens 安装失败？

```bash
# 尝试升级 pip
pip install --upgrade pip

# 从源码安装
git clone https://github.com/jbloomAus/SAELens.git
cd SAELens
pip install -e .
```

### Q2: 与自实现版本结果不同？

这是正常的！原因：
- SAELens 可能使用不同的初始化
- 训练循环实现细节不同
- 归一化和优化策略差异

关键是看是否找到了稀疏的空间特征。

### Q3: 如何使用 WandB 追踪？

```python
# 在配置中启用
cfg = LanguageModelSAERunnerConfig(
    ...
    log_to_wandb=True,
    wandb_project="spatial-reasoning-sae",
    wandb_entity="your-username",
)
```

### Q4: 如何使用其他 SAE 架构？

```python
# Gated SAE (通常更稳定)
architecture="gated"

# Top-K SAE (固定稀疏度)
architecture="topk"
k_sparse=50
```

### Q5: Ablation 实验没有显著效果？

**可能原因：**
1. Spatial features 识别不准确
   - 解决：降低 `CORR_THRESHOLD`，识别更多特征
   - 检查 `dimension_features.json` 的质量

2. Baseline 性能太低（<50%）
   - 解决：检查评估数据格式
   - 确认 `ANSWER_TOKENS` 正确映射

3. 特征数量太少（<10）
   - 解决：增加 SAE 特征数（`--n_features 4096`）
   - 降低 L1 系数（`--l1_coeff 5e-5`）

**调试步骤：**
```bash
# 1. 检查 spatial features 数量
cat sae_results/*/analysis/dimension_features.json | grep "feature_idx"

# 2. 查看详细错误示例
python -c "
import json
with open('sae_results/*/ablation_results/ablation_results_*.json') as f:
    data = json.load(f)
    for ex in data['examples']['spatial_ablation'][:5]:
        if not ex['correct']:
            print(ex)
"
```

### Q6: 运行太慢怎么办？

**优化方案：**
1. 使用 `simple` 版本（快 5-10x）
2. 减少评估样本：`--max_samples 50`
3. 使用更快的 GPU
4. Batch 评估（修改代码）

## 📚 参考资料

1. **SAELens 官方文档**
   - GitHub: https://github.com/jbloomAus/SAELens
   - Docs: https://jbloomaus.github.io/SAELens/

2. **SAE 论文**
   - Anthropic: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features)

3. **相关项目**
   - 自实现版本: `../sae/`
   - Probe 基线: `../probe/`

## 🔄 版本历史

- **v1.0** (2026-01-02): 初始版本，基于 SAELens 3.0+

## ✅ 检查清单

实验前:
- [ ] SAELens 已安装: `pip list | grep sae-lens`
- [ ] 数据文件存在
- [ ] GPU 可用
- [ ] 磁盘空间充足 (>10GB)

实验后:
- [ ] L0 在合理范围 (20-200)
- [ ] 找到稀疏空间特征 (>15)
- [ ] Probe R² > 0 (正数)
- [ ] 可视化已生成

## 🔬 完整实验流程（推荐）

### 快速版（用于测试）

```bash
# Step 1: 训练 SAE（5-10分钟）
python train_sae_saelens.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr "../dataGenerate/spatial_reasoning_dataset_ZH.json" \
  -te "../dataGenerate/spatial_reasoning_dataset_ZH_test.json" \
  --layer 8 --n_features 2048 --max_samples 1000

# Step 2: 分析特征（2-3分钟）
python analyze_features_saelens.py -c "./sae_results/L8_F2048_*/"

# Step 3: 干预实验（5-10分钟）
bash run_ablation.sh ./sae_results/L8_F2048_*/ 100 simple

# Step 4: 可视化（1分钟）
python visualize_ablation.py \
  -r "./sae_results/L8_F2048_*/ablation_results/ablation_results_*.json"
```

**总计：15-25分钟** ✨

### 完整版（用于论文）

```bash
# Step 1: 训练 SAE（30-60分钟）
python train_sae_saelens.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr "../dataGenerate/spatial_reasoning_dataset_ZH.json" \
  -te "../dataGenerate/spatial_reasoning_dataset_ZH_test.json" \
  --layer 8 \
  --n_features 4096 \
  --l1_coeff 5e-5 \
  --num_tokens 500000

# Step 2: 分析特征（5-10分钟）
python analyze_features_saelens.py \
  -c "./sae_results/L8_F4096_*/" \
  --top_k 100

# Step 3: 干预实验 - 全量评估（20-30分钟）
python ablate_features_simple.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c "./sae_results/L8_F4096_*/" \
  -e "../dataGenerate/spatial_reasoning_dataset_ZH_test.json"

# Step 4: 可视化（2分钟）
python visualize_ablation.py \
  -r "./sae_results/L8_F4096_*/ablation_results/ablation_results_*.json"
```

**总计：1-2小时** 🚀

### 实验成功的标志

✅ **训练成功**：
- L0 < 200
- MSE Loss < 0.01
- Dead features < 30%

✅ **分析成功**：
- 找到 > 15 个 sparse spatial features
- 覆盖 5-6 个空间方向
- Probe R² > 0.10

✅ **干预成功**（最重要！）：
- Baseline Acc > 0.70
- Spatial Drop > 10%
- Spatial Drop > 2 × Random Drop

**如果满足以上所有条件 → 实验成功！🎉**

## 🎉 总结

这个基于 SAELens 的实现提供了：
- ✅ 官方库的稳定性和标准化
- ✅ 完整的训练、分析、干预流程
- ✅ 因果验证实验（Ablation）⭐
- ✅ 与研究社区的兼容性
- ✅ 生产级的代码质量

**推荐用于正式实验和论文发表！** 🏆

## 📖 更多文档

- **快速开始**: `QUICKSTART.md`
- **与其他方法对比**: `COMPARISON.md`
- **干预实验详细指南**: `ABLATION_GUIDE.md` ⭐


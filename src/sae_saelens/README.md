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

## 📊 输出结果

### 文件结构

```
sae_results/
└── L8_F2048_L10.0001_YYYYMMDD_HHMMSS/
    ├── sae_checkpoint.pt        # SAE 模型权重
    ├── activations.pt           # 保存的激活和 targets
    ├── training_history.json    # 训练历史
    ├── config.json             # 实验配置
    └── analysis/
        ├── feature_stats.npz              # 特征统计
        ├── feature_target_correlations.npz # 相关性矩阵
        ├── dimension_features.json        # 空间特征分类
        ├── probe_results.json             # Probe 性能
        ├── activation_frequency.png       # 激活频率分布
        └── feature_correlations.png       # 相关性热图
```

### 关键指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **L0** | 20-200 | 平均激活特征数 |
| **Probe R²** | >0.10 | 特征预测能力 |
| **稀疏空间特征** | >15 | 激活率 1-50% 的空间特征 |
| **覆盖维度** | 5-6/6 | 覆盖的空间方向数 |

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

## 🎉 总结

这个基于 SAELens 的实现提供了：
- ✅ 官方库的稳定性和标准化
- ✅ 完整的训练和分析流程
- ✅ 与研究社区的兼容性
- ✅ 生产级的代码质量

**推荐用于正式实验和论文发表！** 🏆


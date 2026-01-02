# SAE 实验：Layer 8 空间推理特征分析

## 实验目标

使用 Sparse Autoencoder (SAE) 分析 Qwen2.5-7B-Instruct Layer 8 是否包含稀疏、可分离、与空间状态强相关的内部特征。

### 具体验证目标

1. **方向/位置选择性 features**: 是否存在对特定空间方向（左/右、上/下、前/后）有强响应的特征
2. **状态保持型 features**: 是否存在维持空间 working memory 的特征
3. **因果作用**: 这些 features 是否对行为有因果影响（后续阶段）

## 实验配置（已优化）

基于 layer sweep probe 结果（Layer 8 R²≈0.27），采用以下配置：

| 项目 | 配置 | 原因 |
|------|------|------|
| 模型 | Qwen2.5-7B-Instruct | 已验证有效 |
| 层 | Layer 8 | Probe 峰值层 |
| Hook点 | mlp_out | 空间/状态特征更稀疏 |
| Token | 最后一个 token | 与 probe 一致，防止答案泄漏 |
| Features | 2048 | 稳妥的第一轮配置 |
| L1系数 | 1e-3 | 中等稀疏度 |

## 代码结构

```
src/sae/
├── train_sae.py           # SAE 训练主脚本
├── analyze_features.py    # 特征分析脚本
├── run_experiment.sh      # 一键运行脚本
└── README.md             # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install sae-lens torch transformers transformer-lens scikit-learn matplotlib seaborn scipy
```

### 2. 训练 SAE

```bash
cd /home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae

python train_sae.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr "../dataGenerate/spatial_reasoning_dataset_ZH.json" \
  -te "../dataGenerate/spatial_reasoning_dataset_ZH_test.json" \
  --layer 8 \
  --n_features 2048 \
  --l1_coeff 1e-3 \
  --batch_size 128 \
  --lr 3e-4 \
  --num_epochs 5 \
  --output_dir "./sae_results"
```

**训练时间**: 约 10-30 分钟（取决于数据量和GPU）

**输出**:
- `sae_results/L8_F2048_L1*_*/sae_checkpoint.pt` - SAE 模型权重
- `sae_results/L8_F2048_L1*_*/training_history.json` - 训练历史
- `sae_results/L8_F2048_L1*_*/training_curves.png` - 训练曲线可视化

### 3. 分析特征

训练完成后，分析学到的特征：

```bash
python analyze_features.py \
  -c "./sae_results/L8_F2048_L1*_*/sae_checkpoint.pt" \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -te "../dataGenerate/spatial_reasoning_dataset_ZH_test.json" \
  --top_k 50
```

**输出**:
- `analysis/feature_stats.npz` - 特征统计
- `analysis/feature_target_correlations.npz` - 特征与空间维度的相关性
- `analysis/top_features.json` - Top-K 最相关特征
- `analysis/dimension_features.json` - 按空间维度分类的特征
- `analysis/probe_results.json` - Feature probe 结果
- `analysis/*.png` - 各种可视化图表

### 4. 使用一键脚本

```bash
bash run_experiment.sh
```

## 输出解读

### 训练阶段关键指标

1. **L0 (Sparsity)**: 平均每个样本激活多少个 features
   - 目标: 20-100 (1-5% 的 2048 features)
   - 太低 (<10): 可能信息丢失
   - 太高 (>200): 不够稀疏，缺少可解释性

2. **MSE (Reconstruction Error)**: 重构误差
   - 越低越好，但需要平衡 L0
   - 典型值: 0.001-0.01

3. **Cosine Similarity**: 重构向量与原始向量的相似度
   - 目标: >0.95
   - <0.9: 重构质量差

### 分析阶段关键发现

1. **Feature-Target Correlation**
   - 查看 `top_features.json` 中相关系数最高的 features
   - |correlation| > 0.2: 强相关
   - |correlation| > 0.15: 中等相关

2. **Dimension Features**
   - 查看 `dimension_features.json`
   - 每个空间维度（左右/上下/前后）应该有专门的 features

3. **Probe R²**
   - 用 SAE features 作为输入训练 probe
   - R² 应该接近或超过原始 mlp_out 的 probe R²
   - Top-K features 的 R² 如果接近全量 features，说明找到了关键特征

## 预期结果

### ✅ 成功的标志

1. **稀疏性**: L0 在 20-100 之间
2. **重构质量**: Cosine Similarity > 0.95
3. **空间相关性**: 至少 30-50 个 features 与空间维度有中等以上相关
4. **Feature Probe**: R² > 0.20 (接近原始 probe 的 0.27)
5. **维度分布**: 六个空间方向都能找到对应的 features

### ⚠️ 需要调整的信号

1. **L0 太高 (>200)**: 增加 l1_coeff (例如 5e-3)
2. **L0 太低 (<10)**: 减少 l1_coeff (例如 5e-4)
3. **重构差 (CosSim<0.9)**: 减少 l1_coeff 或增加 features 数量
4. **找不到空间相关 features**: 
   - 检查数据是否正确
   - 尝试增加训练 epochs
   - 尝试不同的 layer (7-9)

## 下一步计划

1. **特征可视化**: 可视化 top features 激活的样本
2. **因果干预**: 消融特定 features，观察行为变化
3. **Cross-validation**: 在不同数据集上验证 features 的稳定性
4. **扩展到其他层**: 对比不同层的 features

## 常见问题

### Q: 训练很慢？
A: 
- 减少 `--max_samples` (例如 10000)
- 增加 `--batch_size`
- 使用更强的 GPU

### Q: GPU 内存不足？
A: 
- 减少 `--batch_size`
- 使用 `torch.float16` (代码中已默认)
- 减少 `--n_features`

### Q: 找不到明显的空间 features？
A:
1. 检查 probe 是否正确 (应该有 R²~0.27)
2. 尝试调整 l1_coeff
3. 增加训练时间 (10 epochs)
4. 尝试 `hook_resid_post` 而不是 `hook_mlp_out`

### Q: 如何选择最佳的 l1_coeff？
A: 运行 grid search:
```bash
for l1 in 5e-4 1e-3 5e-3; do
  python train_sae.py ... --l1_coeff $l1
done
```
选择 L0 和重构质量平衡最好的。

## 参考文献

- Sparse Autoencoders: [Anthropic Research](https://transformer-circuits.pub/2023/monosemantic-features)
- SAELens: https://github.com/jbloomAus/SAELens
- Layer Sweep Probe: 本项目 `src/probe/`

## 联系

如有问题或发现，请记录在实验日志中。

# SAE 训练问题诊断与解决方案

## 📊 当前问题总结

您的 SAE 训练出现了严重的**特征死亡（Feature Death）**问题：

- **Total features**: 16,384
- **Active features**: 161 (仅 0.98%)
- **Dead features**: 16,192 (98.8%)
- **Probe R²**: -0.0083 (负值！)
- **Spatial features**: 仅 2 个 (1 Left, 1 Above)

这意味着：
- ❌ 模型没有学到有意义的空间表征
- ❌ 绝大多数特征被 L1 正则化"杀死"
- ❌ 线性探针表现比随机猜测还差

---

## 🔍 根本原因分析

### 1. L1 系数过高导致过度稀疏化

**当前配置**:
```bash
L1_COEFF=1e-5  # 太高了！
N_FEATURES=16384
```

**训练历史显示**:
```
Epoch 1:  L0 = 6975  (42.6% 特征激活)
Epoch 10: L0 = 157   (0.96% 特征激活) ❌
```

**解释**:
- L1 惩罚太强，导致特征激活被强制压到接近 0
- 大部分特征权重在训练过程中被"杀死"
- 最终只有 ~157 个特征在工作，其余完全不激活

### 2. 特征数量与数据量不匹配

**数据统计**:
```
训练样本: 2,000
特征数量: 16,384
比例: 0.12 samples/feature  ❌
```

**问题**:
- 每个特征平均只能看到 0.12 个训练样本
- 没有足够数据让这么多特征学习有意义的模式
- 结果：大量特征从未得到有效训练

### 3. 只使用最后一个 Token

**代码问题** (`train_sae_saelens.py:296`):
```python
mlp_out = cache[f"blocks.{layer_idx}.hook_mlp_out"][0, -1]  # 只取最后一个token
```

**影响**:
- 空间信息可能分布在多个 token 位置
- 只用最后一个 token 会丢失大量信息
- 降低了数据利用效率

### 4. Batch Size 太小

**当前配置**:
```python
batch_size = 32  # 太小
n_epochs = 10
```

**问题**:
- 小 batch size 导致训练不稳定
- 对于 16,384 个特征，需要更大的 batch size 来提供稳定的梯度

---

## ✅ 解决方案

### 方案 1: 降低 L1 系数 + 减少特征数（推荐）

**新配置**:
```bash
L1_COEFF=1e-6    # 从 1e-5 降到 1e-6 (降低 10倍)
N_FEATURES=8192  # 从 16384 降到 8192 (减少一半)
```

**原理**:
- 更低的 L1 系数允许更多特征激活
- 更少的特征 + 相同数据 = 更好的训练
- 目标：激活率 5-20%，而非 0.96%

**运行命令**:
```bash
bash run_complete_pipeline.sh --all --l1 1e-6 --features 8192
```

**预期结果**:
- Active features: 400-1600 (5-20%)
- Probe R²: > 0.3
- Spatial features: 10-50 个

---

### 方案 2: 使用更小的字典 + 更激进的 L1 降低

**新配置**:
```bash
L1_COEFF=5e-7     # 更低的 L1
N_FEATURES=4096   # 更少的特征
```

**适用场景**:
- 如果方案 1 仍然特征太稀疏
- 数据量确实有限（2000 samples）

**运行命令**:
```bash
bash run_complete_pipeline.sh --all --l1 5e-7 --features 4096
```

---

### 方案 3: 使用多个 Token 位置（最优）

**修改训练代码**:

修改 `train_sae_saelens.py` 的 `collect_activations_for_saelens` 函数：

```python
def collect_activations_for_saelens(
    model: HookedTransformer,
    data: List[Dict],
    layer_idx: int,
    max_tokens: int = None,
    use_all_tokens: bool = True,  # 新增参数
) -> torch.Tensor:
    """
    收集激活用于 SAELens 训练
    """
    activations = []
    total_tokens = 0
    
    for sample in tqdm(data, desc=f"Collecting activations from Layer {layer_idx}"):
        if max_tokens and total_tokens >= max_tokens:
            break
            
        prompt = sample["prompt"]
        tokens = model.to_tokens(prompt, truncate=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_mlp_out"
            )
        
        mlp_out = cache[f"blocks.{layer_idx}.hook_mlp_out"][0]  # [seq_len, d_mlp]
        
        if use_all_tokens:
            # 使用所有 token（增加数据量）
            for pos in range(mlp_out.shape[0]):
                activations.append(mlp_out[pos].float().cpu())
                total_tokens += 1
                if max_tokens and total_tokens >= max_tokens:
                    break
        else:
            # 只用最后一个 token（原方案）
            activations.append(mlp_out[-1].float().cpu())
            total_tokens += tokens.shape[1]
    
    return torch.stack(activations)
```

**优点**:
- 数据量增加 10-50 倍（取决于序列长度）
- 2,000 samples × 平均 20 tokens = 40,000 训练点
- 每个特征能看到更多样本

**运行**:
需要手动修改代码后运行。

---

### 方案 4: 调整训练超参数

**建议修改** (`train_sae_saelens.py`):
```python
batch_size = 128      # 从 32 增加到 128
n_epochs = 20         # 从 10 增加到 20
warm_up_steps = 500   # 从 100 增加到 500
```

**配合其他方案使用**，单独使用效果有限。

---

## 📈 如何评估改进效果

### 好的训练应该满足:

1. **稀疏性适中**:
   - L0 (active features): 5-20% 的总特征数
   - 例如: 8192 特征 → 400-1600 激活

2. **重建质量**:
   - Test MSE < 0.005
   - MSE 不应该比 L1 loss 大太多

3. **特征质量**:
   - Active features (>1%): > 500
   - Dead features: < 50%
   - Probe R² > 0.3 (理想 > 0.5)

4. **空间特征**:
   - 每个方向至少 5-10 个特征
   - 稀疏空间特征 > 20 个

### 监控指标:

训练时观察:
```python
# 好的训练曲线
Epoch 1:  L0 = 3000-5000  (合理的初始激活)
Epoch 10: L0 = 800-1600   (稳定在 10-20%)
Epoch 20: L0 = 500-1200   (略微下降但不崩溃)
```

---

## 🚀 快速开始

### Step 1: 运行改进的配置

```bash
cd /home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae_new/task_family_1

# 方案 1 (推荐先尝试)
bash run_complete_pipeline.sh --all --l1 1e-6 --features 8192

# 如果还是太稀疏，尝试方案 2
bash run_complete_pipeline.sh --all --l1 5e-7 --features 4096

# 如果要尝试更低的 L1
bash run_complete_pipeline.sh --all --l1 3e-7 --features 8192
```

### Step 2: 查看结果

```bash
# 最新的 checkpoint
CKPT=$(ls -td sae_results_new/L18_* | head -1)

# 查看训练历史
cat $CKPT/training_history.json | grep -A 5 "test_l0"

# 查看分析结果
cat $CKPT/analysis/dimension_features.json | head -50

# 查看关键指标
tail $CKPT/analysis/*.txt
```

### Step 3: 对比不同配置

可以运行多个实验进行对比：

```bash
# 实验 1
bash run_complete_pipeline.sh --train --analyze --l1 1e-6 --features 8192

# 实验 2  
bash run_complete_pipeline.sh --train --analyze --l1 5e-7 --features 8192

# 实验 3
bash run_complete_pipeline.sh --train --analyze --l1 1e-6 --features 4096
```

---

## 🔬 理论背景

### L1 正则化的权衡

SAE 的目标是:
```
Loss = MSE(x, decoder(encoder(x))) + λ * ||encoder(x)||₁
```

- **λ 太大**: 强制稀疏 → 特征死亡 → 信息丢失
- **λ 太小**: 特征不稀疏 → 难以解释 → 失去稀疏性优势
- **λ 合适**: 稀疏但信息丰富 → 可解释的特征

### 特征数量的选择

根据经验法则:
```
n_features / d_model ∈ [2, 8]

对于 d_mlp = 3584:
- 最小: 3584 × 2 = 7,168
- 最大: 3584 × 8 = 28,672
- 推荐: 3584 × 4 = 14,336  或  3584 × 2 = 7,168
```

但要考虑数据量:
```
samples_per_feature = n_samples / n_features > 0.5  (最好 > 2)

当前: 2000 / 16384 = 0.12 ❌
改进: 2000 / 8192 = 0.24  (勉强)
     2000 / 4096 = 0.49  (接近)
     
如果用所有 tokens: 40000 / 8192 = 4.9 ✓
```

---

## 📚 参考文献

1. [Anthropic - Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features)
2. [SAELens Documentation](https://github.com/jbloomAus/SAELens)
3. [OpenAI - Scaling Sparse Autoencoders](https://arxiv.org/abs/2406.04093)

---

## 💡 额外建议

1. **Layer 选择**: 
   - Layer 18 可能不是最优的空间推理层
   - 建议尝试 Layer 12-20 的范围

2. **数据增强**:
   - 如果可能，收集更多训练数据
   - 或者使用数据增强技术

3. **渐进式训练**:
   - 先用低 L1 训练 (1e-6)
   - 再逐渐增加 L1 来提高稀疏性
   - 避免一开始就"杀死"太多特征

4. **不同层的 SAE**:
   - 可以同时训练多个层的 SAE
   - 找到最适合空间推理的层

---

生成时间: 2026-01-05
实验目录: `/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae_new/task_family_1`


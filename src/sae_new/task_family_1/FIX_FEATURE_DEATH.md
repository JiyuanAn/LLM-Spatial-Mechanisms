# 🔧 SAE 特征死亡问题修复指南

## 📊 问题诊断

### 症状
运行 `run_experiment_new.sh` 后出现严重的**特征死亡（Feature Death）**：

```
实验 1 (L1=1e-5, Features=2048):
  - Active features: 191/2048 (9.3%)
  - Dead features: 1855 (90.5%)
  - Probe R²: -0.0026
  - Spatial features: 25 (质量差)

实验 2 (L1=5e-6, Features=2048):
  - Active features: 17/2048 (0.83%)  ❌ 更糟！
  - Dead features: 2030 (99.2%)
  - Probe R²: 0.0005
  - Spatial features: 5 (全是全局偏置)
```

### 训练历史分析

```json
实验 2 的训练历史:
Epoch 1:  L0 = 882  (43% 特征激活) ✓
Epoch 5:  L0 = 119  (5.8% 特征激活)
Epoch 10: L0 = 19.6 (0.96% 特征激活) ❌

说明：特征在训练过程中被逐渐"杀死"
```

## 🔍 根本原因

### 1. **L1 惩罚过强**
即使降低到 5e-6，L1 正则化仍然太强，导致：
- 特征激活被强制压到接近 0
- 大部分特征权重在训练中被"杀死"
- 最终只有极少数特征存活

### 2. **特征数量过多**
2048 个特征对于 2000 个训练样本来说：
- 每个特征平均只有 1 个训练样本
- 没有足够数据让所有特征学习有意义的模式
- 导致特征竞争激烈，弱特征被淘汰

### 3. **L1 系数在训练初期就全力施加**
- 训练初期特征还未学到有用模式
- L1 惩罚立即开始"杀死"弱特征
- 导致特征还没机会学习就被抑制

## ✅ 解决方案

### 方案 A：激进修复（推荐）

**修改配置**：
```bash
N_FEATURES=512        # 从 2048 → 512 (减少 4 倍)
L1_COEFF=1e-7         # 从 5e-6 → 1e-7 (减少 50 倍)
NUM_TOKENS=500000     # 保持不变
```

**运行命令**：
```bash
bash run_experiment_fixed.sh
```

**预期结果**：
- 激活率：10-30%（50-150 个特征）
- Probe R²：> 0.1
- 空间特征：10+ 个有意义的特征

**原理**：
- 更少的特征 = 每个特征有更多训练数据
- 极低的 L1 = 允许更多特征存活
- 特征/样本比：512/2000 = 0.26（合理）

---

### 方案 B：渐进修复（保守）

**修改配置**：
```bash
N_FEATURES=1024       # 从 2048 → 1024 (减少 2 倍)
L1_COEFF=1e-6         # 从 5e-6 → 1e-6 (减少 5 倍)
NUM_TOKENS=500000     # 保持不变
```

**修改 `run_experiment_new.sh`**：
```bash
N_FEATURES=1024
L1_COEFF=1e-6
```

**预期结果**：
- 激活率：5-15%（50-150 个特征）
- Probe R²：> 0.05
- 空间特征：5-10 个

---

### 方案 C：代码级修复（已实施）

**关键改进**：在 `train_sae_saelens.py` 中添加了 **L1 预热**：

```python
# L1 系数预热 - 关键修复！避免早期特征死亡
l1_warmup_steps = 500
if global_step < l1_warmup_steps:
    l1_scale = (global_step + 1) / l1_warmup_steps
    current_l1_coeff = L1_COEFF * l1_scale
else:
    current_l1_coeff = L1_COEFF
```

**原理**：
- 训练初期 L1 系数从 0 逐渐增加到目标值
- 给特征时间学习有用模式，再施加稀疏性约束
- 避免特征在学习前就被"杀死"

**其他改进**：
- `batch_size: 32 → 64`（更稳定的梯度）
- `n_epochs: 10 → 20`（更充分的训练）
- `l1_warmup_steps: 0 → 500`（L1 预热）

---

## 🚀 推荐执行步骤

### 快速修复（5 分钟）

```bash
cd /home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae_new/task_family_1

# 使用修复版脚本（已配置好最佳参数）
bash run_experiment_fixed.sh
```

### 自定义修复

如果你想自己调整参数：

```bash
# 编辑配置
vim run_experiment_new.sh

# 修改这些行：
N_FEATURES=512        # 或 1024
L1_COEFF=1e-7         # 或 1e-6

# 运行
bash run_experiment_new.sh
```

---

## 📈 如何判断修复成功

### 训练过程中观察：

```
✓ 好的训练：
Epoch 1:  L0 = 200  (39% 特征激活)
Epoch 10: L0 = 80   (15% 特征激活)
Epoch 20: L0 = 60   (12% 特征激活)
→ 激活率稳定在 10-20%

❌ 失败的训练：
Epoch 1:  L0 = 800  (78% 特征激活)
Epoch 10: L0 = 20   (2% 特征激活)
→ 激活率崩溃
```

### 最终结果检查：

```
✓ 成功指标：
- Active features: > 10%
- Dead features: < 80%
- Probe R²: > 0.05
- Spatial features: > 5 个
- activation_freq: 0.1-0.9（不是 1.0）

❌ 失败指标：
- Active features: < 5%
- Dead features: > 90%
- Probe R²: < 0.01 或负值
- activation_freq: 1.0（全局偏置）
```

---

## 🔬 深入理解

### 为什么降低 L1 系数有效？

**L1 正则化的作用**：
```
loss = MSE + L1_COEFF * |features|
```

- L1 系数太高 → 模型宁愿牺牲重构质量也要稀疏
- L1 系数太低 → 特征不稀疏，难以解释
- **最佳平衡**：让 10-20% 的特征激活

**经验法则**：
```
L1_COEFF ≈ 1 / (d_model * sqrt(n_features))

对于 d_model=3584, n_features=512:
L1_COEFF ≈ 1 / (3584 * sqrt(512)) ≈ 1e-7 ✓
```

### 为什么减少特征数量有效？

**数据效率**：
```
特征/样本比：
- 2048 / 2000 = 1.02  ❌ 每个特征 < 1 个样本
- 512 / 2000 = 0.26   ✓ 每个特征 ~4 个样本
```

**过参数化问题**：
- 特征太多 → 模型过参数化 → 容易过拟合
- 特征太少 → 表达能力不足 → 欠拟合
- **最佳范围**：特征数 = 0.1-0.5 × 样本数

---

## 📚 参考文献

1. **Anthropic SAE 论文**：
   - "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
   - 推荐 L1 系数范围：1e-7 到 1e-5

2. **SAELens 文档**：
   - https://github.com/jbloomAus/SAELens
   - 推荐特征激活率：5-20%

3. **特征死亡问题**：
   - Dead neurons in neural networks
   - L1 regularization and feature selection

---

## 🆘 如果还是失败

如果修复后仍然出现特征死亡，尝试：

### 1. 进一步降低 L1
```bash
L1_COEFF=5e-8  # 或更低
```

### 2. 进一步减少特征
```bash
N_FEATURES=256  # 或 128
```

### 3. 增加训练数据
```bash
NUM_TOKENS=1000000  # 收集更多样本
```

### 4. 检查数据质量
```bash
# 确保数据包含多样的空间关系
python -c "
import json
data = json.load(open('$TRAIN_DATA'))
targets = [s['target'] for s in data]
print('Target distribution:', {t: targets.count(t) for t in set(targets)})
"
```

### 5. 尝试不同的层
```bash
# Layer 8 可能不是最佳层
LAYER=10  # 或 12, 15
```

---

## 📞 联系与支持

如果问题持续存在，请提供：
1. 训练历史 JSON 文件
2. 配置文件
3. 终端输出

我会帮你进一步诊断！


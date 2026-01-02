# SAE Feature Ablation 使用示例

## 实际运行示例

假设你已经完成了 SAE 训练和特征分析，现在要运行 ablation 实验。

### 场景 1: 快速测试（5分钟）

你想快速验证代码是否正常工作：

```bash
cd /home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae_saelens

# 找到最新的 SAE checkpoint
SAE_DIR=$(ls -td sae_results/L8_* | head -1)
echo "Using SAE: $SAE_DIR"

# 运行 ablation（仅 50 个样本）
python ablate_features_simple.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --sae_checkpoint "$SAE_DIR" \
    --eval_data_file "../../data/spatial/test_data.json" \
    --max_samples 50
```

**预期输出：**
```
============================================================
Loading SAE checkpoint...
============================================================
SAE: Layer 8, 3584 -> 2048 features

============================================================
Loading spatial features...
============================================================
Loaded 42 spatial features

============================================================
Running Ablation Experiments
============================================================

Conditions:
  1. No ablation (baseline)
  2. Spatial feature ablation (42 features)
  3. Random feature ablation (42 features)

Condition 1: No ablation...
No ablation: 100%|████████████████| 50/50 [00:15<00:00,  3.21it/s]
  Accuracy: 0.8400 (42/50)

Condition 2: Spatial feature ablation...
Spatial ablation: 100%|██████████| 50/50 [00:15<00:00,  3.18it/s]
  Accuracy: 0.6200 (31/50)

Condition 3: Random feature ablation...
Random ablation: 100%|███████████| 50/50 [00:15<00:00,  3.19it/s]
  Accuracy: 0.8000 (40/50)

============================================================
RESULTS
============================================================

Accuracy:
  Baseline (no ablation)   : 0.8400
  Spatial ablation         : 0.6200 (Δ = -0.2200)
  Random ablation          : 0.8000 (Δ = -0.0400)

Relative drop:
  Spatial ablation         : 26.19%
  Random ablation          : 4.76%

Effect size (spatial - random): -0.2200

============================================================
CONCLUSION
============================================================
✓ Spatial features have causal importance!
  Ablating spatial features hurts performance more than random features.
  Effect is substantial (>5% accuracy drop).
```

### 场景 2: 论文级别的完整实验（30分钟）

```bash
# 使用更多样本，获得更可靠的结果
python ablate_features_simple.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --sae_checkpoint "./sae_results/L8_F2048_20260102_120000" \
    --eval_data_file "../../data/spatial/test_data.json" \
    --max_samples 500 \
    --seed 42

# 可视化结果
python visualize_ablation.py \
    --results_file "./sae_results/L8_F2048_20260102_120000/ablation_results/ablation_results_*.json"
```

**生成的文件：**
```
ablation_results/
├── ablation_results_20260102_150530.json
├── ablation_summary_20260102_150530.txt
├── ablation_accuracy_bar.png
├── ablation_drop.png
├── ablation_combined.png
├── ablation_flips.png
└── ablation_table.tex
```

### 场景 3: 多个随机种子重复实验

为了确保结果的稳定性：

```bash
#!/bin/bash
# repeat_ablation.sh

SAE_DIR="./sae_results/L8_F2048_20260102_120000"
EVAL_DATA="../../data/spatial/test_data.json"

for seed in 42 43 44 45 46; do
    echo "Running with seed $seed..."
    
    python ablate_features_simple.py \
        --model_name "Qwen/Qwen2.5-7B-Instruct" \
        --sae_checkpoint "$SAE_DIR" \
        --eval_data_file "$EVAL_DATA" \
        --max_samples 200 \
        --seed $seed
    
    echo "Seed $seed complete!"
    echo "---"
done

# 分析结果
python -c "
import json
import glob
import numpy as np

results_files = glob.glob('$SAE_DIR/ablation_results/ablation_results_*.json')

spatial_drops = []
random_drops = []

for f in results_files:
    with open(f) as fp:
        data = json.load(fp)
        spatial_drops.append(data['results']['spatial_ablation']['drop_pct'])
        random_drops.append(data['results']['random_ablation']['drop_pct'])

print('Spatial Drop: {:.2f} ± {:.2f}%'.format(
    np.mean(spatial_drops), np.std(spatial_drops)))
print('Random Drop: {:.2f} ± {:.2f}%'.format(
    np.mean(random_drops), np.std(random_drops)))
"
```

### 场景 4: 分维度分析

只测试特定空间维度的特征：

```python
# analyze_by_dimension.py
import json
import torch
from pathlib import Path

# 加载 dimension features
sae_dir = Path("./sae_results/L8_F2048_20260102_120000")
with open(sae_dir / "analysis/dimension_features.json") as f:
    dim_features = json.load(f)

# 测试 X 维度（left/right）
x_features = [f['feature_idx'] for f in dim_features['X_positive']]
x_features += [f['feature_idx'] for f in dim_features['X_negative']]
print(f"X-axis features: {len(x_features)}")

# 测试 Y 维度（above/below）
y_features = [f['feature_idx'] for f in dim_features['Y_positive']]
y_features += [f['feature_idx'] for f in dim_features['Y_negative']]
print(f"Y-axis features: {len(y_features)}")

# 测试 Z 维度（front/behind）
z_features = [f['feature_idx'] for f in dim_features['Z_positive']]
z_features += [f['feature_idx'] for f in dim_features['Z_negative']]
print(f"Z-axis features: {len(z_features)}")

# 保存用于后续分析
torch.save({
    'x_features': x_features,
    'y_features': y_features,
    'z_features': z_features,
}, sae_dir / "dimension_feature_ids.pt")
```

然后修改 `ablate_features_simple.py` 中的 `spatial_feature_ids` 来测试特定维度。

## 常见错误和解决方案

### 错误 1: 找不到 config.py

```
ModuleNotFoundError: No module named 'config'
```

**解决方案：**
```bash
# 检查 config.py 位置
ls ../../config.py

# 如果不存在，创建一个：
cat > ../../config.py << 'EOF'
PATHS = {
    "Qwen/Qwen2.5-7B-Instruct": "/path/to/your/Qwen2.5-7B-Instruct",
}
EOF
```

### 错误 2: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**解决方案：**
1. 减少样本数：`--max_samples 50`
2. 使用 CPU（慢）：`--device cpu`
3. 清理 GPU 缓存：
```python
import torch
torch.cuda.empty_cache()
```

### 错误 3: dimension_features.json 不存在

```
Error: .../analysis/dimension_features.json not found!
```

**解决方案：**
```bash
# 先运行特征分析
python analyze_features_saelens.py \
    -c "./sae_results/L8_F2048_*/"
```

### 错误 4: Baseline accuracy 太低

```
Baseline (no ablation)   : 0.1500
```

**原因：** `ANSWER_TOKENS` 映射错误

**解决方案：**
```python
# 检查 token IDs
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/Qwen2.5-7B-Instruct")

for word in ['left', 'right', 'above', 'below', 'front', 'behind']:
    tokens = tokenizer.encode(word, add_special_tokens=False)
    print(f"{word}: {tokens}")
```

## 结果解读

### 示例 1: 强因果证据

```
Baseline           : 0.8500
Spatial ablation   : 0.5800 (-0.2700)
Random ablation    : 0.8200 (-0.0300)

Relative drop:
  Spatial: 31.76%
  Random: 3.53%
```

**解读：** ✅ **非常好！** Spatial features 对空间推理有强因果作用。

### 示例 2: 弱因果证据

```
Baseline           : 0.7800
Spatial ablation   : 0.7000 (-0.0800)
Random ablation    : 0.7400 (-0.0400)

Relative drop:
  Spatial: 10.26%
  Random: 5.13%
```

**解读：** ~ **一般。** Spatial features 有一定作用，但不是唯一重要的。

### 示例 3: 无因果证据

```
Baseline           : 0.6500
Spatial ablation   : 0.6300 (-0.0200)
Random ablation    : 0.6200 (-0.0300)

Relative drop:
  Spatial: 3.08%
  Random: 4.62%
```

**解读：** ✗ **失败。** Spatial features 可能只是相关，不是因果必要的。

**下一步：**
- 增加 SAE 特征数（`--n_features 4096`）
- 降低 L1 系数（`--l1_coeff 5e-5`）
- 尝试其他 layer

## 快速检查清单

在运行 ablation 之前：

- [ ] SAE 已训练完成
- [ ] `sae_checkpoint.pt` 存在
- [ ] 已运行 `analyze_features_saelens.py`
- [ ] `analysis/dimension_features.json` 存在
- [ ] Spatial features > 10
- [ ] 评估数据文件存在
- [ ] GPU 内存充足

运行后检查：

- [ ] Baseline accuracy > 0.60
- [ ] Spatial drop > Random drop
- [ ] 结果文件已生成
- [ ] 可视化图表已生成

## 下一步

1. **写论文：** 使用生成的图表和 LaTeX 表格
2. **深入分析：** 查看哪些样本最受影响
3. **扩展实验：** 尝试其他 layers 或模型
4. **开源分享：** 将结果和代码分享给社区

## 参考

- 详细指南：`ABLATION_GUIDE.md`
- 完整文档：`README.md`
- 快速开始：`QUICKSTART.md`


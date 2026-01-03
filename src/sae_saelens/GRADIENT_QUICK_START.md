# Gradient Attribution 快速开始

## 🚀 一分钟快速开始

```bash
cd /home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/sae_saelens

# 给脚本添加执行权限
chmod +x run_gradient_attribution.sh

# 运行（假设你已有 SAE checkpoint）
bash run_gradient_attribution.sh ./sae_results_new/L8_F2048_*/ 50 grad_x_act
```

## 📊 三种归因方法对比

| 方法 | 速度 | 精度 | 推荐场景 |
|------|------|------|----------|
| `grad_x_act` | ⚡⚡⚡ 快 | ⭐⭐⭐ 好 | **默认选择** |
| `grad_norm` | ⚡⚡⚡ 快 | ⭐⭐ 中 | 诊断用途 |
| `integrated_gradients` | ⚡ 慢 | ⭐⭐⭐⭐ 很好 | 精确分析 |

## 🎯 完整工作流

```bash
# 1. 训练 SAE
python train_sae_saelens_new.py ...

# 2. 分析特征
python analyze_features_saelens_new.py -c ./sae_results_new/L8_F2048_*/

# 3. Gradient 归因（NEW!）
bash run_gradient_attribution.sh ./sae_results_new/L8_F2048_*/ 500 grad_x_act

# 4. Ablation 验证
bash run_ablation_new.sh ./sae_results_new/L8_F2048_*/ 100 simple

# 5. 对比方法
python compare_methods.py \
  -g ./sae_results_new/L8_F2048_*/gradient_attribution/gradient_attribution_*.json \
  -a ./sae_results_new/L8_F2048_*/ablation_results/ablation_results_*.json
```

## 📈 关键指标

- **Spatial/Non-Spatial Ratio > 1.5x**：Spatial features 更重要 ✅
- **Top-K Overlap > 60%**：与已识别 spatial features 一致 ✅
- **Accuracy > 0.70**：模型性能正常 ✅

## 💡 优势

- ⚡ **速度快**：比 ablation 快 10-50 倍
- 📊 **全特征分析**：可以同时分析所有特征
- 🎯 **连续分数**：提供细粒度的重要性分数
- 🔄 **可组合**：与 ablation 方法互补

更多详情请查看 `GRADIENT_GUIDE.md`



"""
诊断脚本：分析两次运行结果差异的原因
"""
import json
import numpy as np

# 加载两个结果文件
result1_path = "probe_procedure_results_20260104_172938.json"  # Com3
result2_path = "probe_procedure_results_20260104_174010.json"  # Com6

with open(result1_path, 'r') as f:
    result1 = json.load(f)

with open(result2_path, 'r') as f:
    result2 = json.load(f)

print("=" * 70)
print("诊断报告：两次运行结果差异分析")
print("=" * 70)

print("\n1. 基本信息对比：")
print(f"   Com3 (计算节点3): Best Layer = {result1['best_layer']}, Best R² = {result1['layer_r2'][result1['best_layer']]:.4f}")
print(f"   Com6 (计算节点6): Best Layer = {result2['best_layer']}, Best R² = {result2['layer_r2'][result2['best_layer']]:.4f}")
print(f"   模型层数: {result1['n_layers']} vs {result2['n_layers']}")
print(f"   模型维度: {result1['d_model']} vs {result2['d_model']}")

print("\n2. 性能指标对比（前10层）：")
print(f"{'Layer':<8} {'Com3 R²':<12} {'Com6 R²':<12} {'差异':<12}")
print("-" * 50)
for i in range(min(10, result1['n_layers'])):
    diff = result2['layer_r2'][i] - result1['layer_r2'][i]
    print(f"{i:<8} {result1['layer_r2'][i]:<12.4f} {result2['layer_r2'][i]:<12.4f} {diff:<12.4f}")

print("\n3. 统计特征：")
r2_1 = np.array(result1['layer_r2'])
r2_2 = np.array(result2['layer_r2'])
mae_1 = np.array(result1['layer_mae'])
mae_2 = np.array(result2['layer_mae'])
rmse_1 = np.array(result1['layer_rmse'])
rmse_2 = np.array(result2['layer_rmse'])

print(f"   Com3 - R² 范围: [{r2_1.min():.4f}, {r2_1.max():.4f}], 均值: {r2_1.mean():.4f}")
print(f"   Com6 - R² 范围: [{r2_2.min():.4f}, {r2_2.max():.4f}], 均值: {r2_2.mean():.4f}")
print(f"   Com3 - MAE 范围: [{mae_1.min():.4f}, {mae_1.max():.4f}], 均值: {mae_1.mean():.4f}")
print(f"   Com6 - MAE 范围: [{mae_2.min():.4f}, {mae_2.max():.4f}], 均值: {mae_2.mean():.4f}")
print(f"   Com3 - RMSE 范围: [{rmse_1.min():.4f}, {rmse_1.max():.4f}], 均值: {rmse_1.mean():.4f}")
print(f"   Com6 - RMSE 范围: [{rmse_2.min():.4f}, {rmse_2.max():.4f}], 均值: {rmse_2.mean():.4f}")

print("\n4. 趋势分析：")
print(f"   Com3 - 正 R² 的层数: {(r2_1 > 0).sum()} / {len(r2_1)}")
print(f"   Com6 - 正 R² 的层数: {(r2_2 > 0).sum()} / {len(r2_2)}")
print(f"   Com3 - R² 趋势: {'递减' if r2_1[-1] < r2_1[0] else '递增'}")
print(f"   Com6 - R² 趋势: {'先增后减（正常）' if r2_2[result2['best_layer']] > r2_2[0] and r2_2[-1] < r2_2[result2['best_layer']] else '异常'}")

print("\n5. 可能的原因分析：")
print("   ✗ Com3 结果异常的特征：")
print("     - 最好的层是第0层（通常应该在中间层）")
print("     - 大部分层 R² 为负（说明预测比均值还差）")
print("     - RMSE 异常高（7-13 vs 5-7）")
print("     - 出现 LinAlgWarning: Ill-conditioned matrix 警告")
print("\n   ✓ Com6 结果正常的特征：")
print("     - 最好的层在第13层（中间层，符合预期）")
print("     - R² 有合理的层间变化模式")
print("     - 性能指标在合理范围内")

print("\n6. 推测的根本原因：")
print("   可能性1: 数据加载问题")
print("     - Com3 可能加载了错误的数据或数据顺序错误")
print("     - 建议：检查数据加载日志，验证样本数量和内容")
print("\n   可能性2: 模型加载问题")
print("     - Com3 可能加载了损坏的模型或缓存")
print("     - 建议：清除模型缓存，重新加载模型")
print("\n   可能性3: 数值稳定性问题")
print("     - Com3 的病态矩阵警告说明存在严重的数值问题")
print("     - 可能原因：Ridge 正则化不足、数据分布异常、特征值崩溃")
print("     - 建议：增加 Ridge alpha 参数，检查特征提取是否正确")
print("\n   可能性4: GPU/硬件差异")
print("     - 不同 GPU 的浮点运算精度可能略有差异")
print("     - 但这不足以解释如此巨大的性能差异")

print("\n7. 建议的调试步骤：")
print("   1. 在同一节点上重新运行，看是否能复现")
print("   2. 添加日志输出数据样本数量和前几个样本的内容")
print("   3. 输出每层提取的特征统计信息（均值、方差、是否有 NaN/Inf）")
print("   4. 增加 Ridge 正则化参数（如从 1.0 改为 10.0）")
print("   5. 检查模型加载时的随机种子设置是否完整")
print("   6. 验证 tokenizer 和模型版本是否一致")

print("\n" + "=" * 70)


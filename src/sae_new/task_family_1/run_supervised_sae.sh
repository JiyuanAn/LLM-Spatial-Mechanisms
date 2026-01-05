#!/bin/bash

# 监督式 SAE 实验 - 通过监督信号保留空间信息
# Supervised SAE Training Script

set -e

PYTHON_BIN="$HOME/.conda/envs/SA/bin/python"

echo "======================================================"
echo "Supervised SAE Training and Analysis Pipeline"
echo "======================================================"
echo ""
echo "核心创新："
echo "  - 在SAE训练中加入空间方向分类任务"
echo "  - 确保特征保留空间信息"
echo "  - 解决无监督SAE信息丢失问题"
echo ""
echo "预期改进："
echo "  - Probe R²: 0.20-0.30 (vs 无监督 0.01)"
echo "  - 空间特征: 30-50 个 (vs 无监督 0个)"
echo "  - 分类准确率: 60-80% (vs 随机 16.7%)"
echo ""

# 配置参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_with_prompt.json"
TEST_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test_with_prompt.json"
OUTPUT_DIR="./sae_results_supervised"

# ============================================
# 使用最佳层和优化后的超参数
# ============================================
LAYER=18              # 从探针结果选择的最佳层 (R²=0.369)
N_FEATURES=512        # 较少的特征数，提高训练质量
L1_COEFF=1e-7         # 极低 L1 避免特征死亡
TASK_COEFF=0.1        # 任务损失权重（新增）
LR=3e-4
NUM_TOKENS=500000
TOP_K=50

MAX_SAMPLES=""

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Layer: $LAYER (best from probe: R²=0.369)"
echo "  Features: $N_FEATURES"
echo "  L1 Coefficient: $L1_COEFF"
echo "  Task Coefficient: $TASK_COEFF (NEW!)"
echo "  Training Tokens: $NUM_TOKENS"
echo ""

# ====================================================
# Step 1: 训练监督式 SAE
# ====================================================
echo "======================================================"
echo "Step 1: Training Supervised SAE"
echo "======================================================"
echo ""
echo "Training with supervision signal..."
echo "Loss = MSE + L1 * $L1_COEFF + Task * $TASK_COEFF"
echo ""

$PYTHON_BIN train_supervised_sae.py \
  -m "$MODEL_NAME" \
  -tr "$TRAIN_DATA" \
  -te "$TEST_DATA" \
  --layer $LAYER \
  --n_features $N_FEATURES \
  --l1_coeff $L1_COEFF \
  --task_coeff $TASK_COEFF \
  --lr $LR \
  --num_tokens $NUM_TOKENS \
  --output_dir "$OUTPUT_DIR" \
  $MAX_SAMPLES

echo ""
echo "✓ Supervised SAE training completed!"
echo ""

# ====================================================
# Step 2: 找到最新的 checkpoint
# ====================================================
echo "======================================================"
echo "Step 2: Locating checkpoint"
echo "======================================================"
echo ""

LATEST_EXP=$(ls -td ${OUTPUT_DIR}/L${LAYER}_F${N_FEATURES}_* 2>/dev/null | head -n1)

if [ -z "$LATEST_EXP" ]; then
    echo "Error: No checkpoint found in $OUTPUT_DIR"
    exit 1
fi

echo "Found checkpoint: $LATEST_EXP"
echo ""

# ====================================================
# Step 3: 分析特征
# ====================================================
echo "======================================================"
echo "Step 3: Analyzing Features"
echo "======================================================"
echo ""

$PYTHON_BIN analyze_features_saelens.py \
  -c "$LATEST_EXP" \
  --top_k $TOP_K

echo ""
echo "✓ Feature analysis completed!"
echo ""

# ====================================================
# Step 4: 对比无监督 vs 监督式
# ====================================================
echo "======================================================"
echo "Step 4: Comparing Results"
echo "======================================================"
echo ""

# 检查是否有无监督版本的结果
UNSUPERVISED_DIR="./sae_results/L${LAYER}_F${N_FEATURES}_L1${L1_COEFF}_*"
if ls -d $UNSUPERVISED_DIR 2>/dev/null | head -1 > /dev/null; then
    echo "Found unsupervised SAE results for comparison:"
    UNSUP_LATEST=$(ls -td $UNSUPERVISED_DIR 2>/dev/null | head -n1)
    echo "  Unsupervised: $UNSUP_LATEST"
    echo "  Supervised:   $LATEST_EXP"
    echo ""
    
    # 读取两者的结果进行对比
    echo "Performance Comparison:"
    echo "  (Check analysis/dimension_features.json for details)"
else
    echo "No unsupervised results found for direct comparison."
    echo "Run run_experiment_L18.sh first to compare."
fi

echo ""

# ====================================================
# Step 5: 总结
# ====================================================
echo "======================================================"
echo "SUPERVISED SAE EXPERIMENT COMPLETED"
echo "======================================================"
echo ""
echo "Results saved to:"
echo "  Training: $LATEST_EXP"
echo "  Analysis: ${LATEST_EXP}/analysis"
echo ""
echo "Key files:"
echo "  - SAE + Classifier: ${LATEST_EXP}/sae_checkpoint.pt"
echo "  - Best model: ${LATEST_EXP}/best_model.pt"
echo "  - Training history: ${LATEST_EXP}/training_history.json"
echo "  - Activations + Labels: ${LATEST_EXP}/activations_with_labels.pt"
echo "  - Feature analysis: ${LATEST_EXP}/analysis/"
echo ""
echo "Expected Improvements over Unsupervised SAE:"
echo "  ✓ Higher Probe R² (0.20-0.30 vs 0.01)"
echo "  ✓ More spatial features (30-50 vs 0)"
echo "  ✓ Better direction coverage (all 6 directions)"
echo "  ✓ Higher classification accuracy (60-80% vs 16.7%)"
echo ""
echo "Next steps:"
echo "  1. Check training history: cat ${LATEST_EXP}/training_history.json"
echo "  2. Review spatial features: cat ${LATEST_EXP}/analysis/dimension_features.json"
echo "  3. Compare with unsupervised: diff results"
echo ""
echo "======================================================"


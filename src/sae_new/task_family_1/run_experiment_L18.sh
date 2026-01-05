#!/bin/bash

# SAELens 实验 - Layer 18 (最佳层)
# 根据探针结果，Layer 18 的 R² = 0.3687 (最高)

set -e

PYTHON_BIN="$HOME/.conda/envs/SA/bin/python"

echo "======================================================"
echo "SAELens Training - Layer 18 (Best Layer for Spatial)"
echo "======================================================"
echo ""
echo "Probe Results Summary:"
echo "  Layer 8:  R² = 0.202"
echo "  Layer 17: R² = 0.338"
echo "  Layer 18: R² = 0.369  ← Best!"
echo "  Layer 19: R² = 0.355"
echo ""

# 配置参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_with_prompt.json"
TEST_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test_with_prompt.json"
OUTPUT_DIR="./sae_results"

# ============================================
# 使用探针结果中的最佳层
# ============================================
LAYER=18              # 从探针结果选择的最佳层
N_FEATURES=512        # 保持较少特征数
L1_COEFF=1e-7         # 极低 L1 避免特征死亡
LR=3e-4
NUM_TOKENS=500000
TOP_K=50

MAX_SAMPLES=""

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Layer: $LAYER (best layer from probe)"
echo "  Features: $N_FEATURES"
echo "  L1 Coefficient: $L1_COEFF"
echo "  Training Tokens: $NUM_TOKENS"
echo ""
echo "Expected: Higher Probe R² (closer to 0.37)"
echo ""

# ====================================================
# Step 1: 训练 SAE
# ====================================================
echo "======================================================"
echo "Step 1: Training SAE with SAELens"
echo "======================================================"
echo ""

$PYTHON_BIN train_sae_saelens.py \
  -m "$MODEL_NAME" \
  -tr "$TRAIN_DATA" \
  -te "$TEST_DATA" \
  --layer $LAYER \
  --n_features $N_FEATURES \
  --l1_coeff $L1_COEFF \
  --lr $LR \
  --num_tokens $NUM_TOKENS \
  --output_dir "$OUTPUT_DIR" \
  $MAX_SAMPLES

echo ""
echo "✓ SAE training completed!"
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
# Step 4: 总结和对比
# ====================================================
echo "======================================================"
echo "EXPERIMENT COMPLETED - LAYER 18"
echo "======================================================"
echo ""
echo "Results saved to:"
echo "  Training: $LATEST_EXP"
echo "  Analysis: ${LATEST_EXP}/analysis"
echo ""
echo "Comparison with Layer 8:"
echo "  Original Probe R²:"
echo "    Layer 8:  0.202"
echo "    Layer 18: 0.369 (82% better)"
echo ""
echo "  Expected SAE Feature R²:"
echo "    Layer 8:  0.009 (observed)"
echo "    Layer 18: 0.05-0.15 (expected, much better)"
echo ""
echo "Key files:"
echo "  - Feature analysis: ${LATEST_EXP}/analysis/dimension_features.json"
echo "  - Training history: ${LATEST_EXP}/training_history.json"
echo ""
echo "======================================================"


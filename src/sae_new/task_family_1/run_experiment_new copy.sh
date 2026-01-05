#!/bin/bash

# SAELens 实验一键运行脚本

set -e

# 使用 SA conda 环境的 Python
PYTHON_BIN="$HOME/.conda/envs/SA/bin/python"

echo "======================================================"
echo "SAELens Training and Analysis Pipeline"
echo "======================================================"
echo ""

# 配置参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_with_prompt.json" #"/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/spatial_reasoning_dataset_EN_with_prompt.json"
TEST_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test_with_prompt.json" #"/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/spatial_reasoning_dataset_EN_test_with_prompt.json"
OUTPUT_DIR="./sae_results_new"
LAYER=8
N_FEATURES=2048 #16384  # 4x 扩展 (d_mlp=3584, expansion=4x)
L1_COEFF=1e-5     # 降低 L1 系数避免特征死亡
LR=3e-4
NUM_TOKENS=100000
TOP_K=50

# 可选：限制样本数量（快速测试）
# MAX_SAMPLES="--max_samples 5000"
MAX_SAMPLES=""

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Layer: $LAYER"
echo "  Features: $N_FEATURES"
echo "  L1 Coefficient: $L1_COEFF"
echo "  Training Tokens: $NUM_TOKENS"
echo ""

# ====================================================
# Step 1: 训练 SAE (SAELens)
# ====================================================
echo "======================================================"
echo "Step 1: Training SAE with SAELens"
echo "======================================================"
echo ""

$PYTHON_BIN train_sae_saelens_new.py \
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

$PYTHON_BIN analyze_features_saelens_new.py \
  -c "$LATEST_EXP" \
  --top_k $TOP_K

echo ""
echo "✓ Feature analysis completed!"
echo ""

# ====================================================
# Step 4: 总结
# ====================================================
echo "======================================================"
echo "EXPERIMENT COMPLETED"
echo "======================================================"
echo ""
echo "Results saved to:"
echo "  Training: $LATEST_EXP"
echo "  Analysis: ${LATEST_EXP}/analysis"
echo ""
echo "Key files:"
echo "  - SAE checkpoint: ${LATEST_EXP}/sae_checkpoint.pt"
echo "  - Training history: ${LATEST_EXP}/training_history.json"
echo "  - Activations: ${LATEST_EXP}/activations.pt"
echo "  - Feature analysis: ${LATEST_EXP}/analysis/"
echo ""
echo "Next steps:"
echo "  1. Review training history: cat ${LATEST_EXP}/training_history.json"
echo "  2. Check spatial features: cat ${LATEST_EXP}/analysis/dimension_features.json"
echo "  3. View visualizations: open ${LATEST_EXP}/analysis/*.png"
echo ""
echo "======================================================"


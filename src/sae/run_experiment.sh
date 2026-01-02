#!/bin/bash

# SAE 实验一键运行脚本
# 使用方法: bash run_experiment.sh

set -e  # 遇到错误立即退出

echo "======================================================"
echo "SAE Training and Analysis Pipeline"
echo "======================================================"
echo ""

# 配置参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="../dataGenerate/spatial_reasoning_dataset_ZH.json"
TEST_DATA="../dataGenerate/spatial_reasoning_dataset_ZH_test.json"
OUTPUT_DIR="./sae_results"
LAYER=8
N_FEATURES=2048
L1_COEFF=1e-4
BATCH_SIZE=128
LR=3e-4
NUM_EPOCHS=5
TOP_K=50

# 可选: 限制训练样本数量（用于快速测试）
# MAX_SAMPLES="--max_samples 5000"
MAX_SAMPLES=""

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Layer: $LAYER"
echo "  Features: $N_FEATURES"
echo "  L1 Coefficient: $L1_COEFF"
echo "  Epochs: $NUM_EPOCHS"
echo ""

# ====================================================
# Step 1: 训练 SAE
# ====================================================
echo "======================================================"
echo "Step 1: Training SAE"
echo "======================================================"
echo ""

python train_sae.py \
  -m "$MODEL_NAME" \
  -tr "$TRAIN_DATA" \
  -te "$TEST_DATA" \
  --layer $LAYER \
  --n_features $N_FEATURES \
  --l1_coeff $L1_COEFF \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --num_epochs $NUM_EPOCHS \
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

# 找到最新创建的实验目录
LATEST_EXP=$(ls -td ${OUTPUT_DIR}/L${LAYER}_F${N_FEATURES}_* 2>/dev/null | head -n1)

if [ -z "$LATEST_EXP" ]; then
    echo "Error: No checkpoint found in $OUTPUT_DIR"
    exit 1
fi

CHECKPOINT="${LATEST_EXP}/sae_checkpoint.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "Found checkpoint: $CHECKPOINT"
echo ""

# ====================================================
# Step 3: 分析特征
# ====================================================
echo "======================================================"
echo "Step 3: Analyzing Features"
echo "======================================================"
echo ""

python analyze_features.py \
  -c "$CHECKPOINT" \
  -m "$MODEL_NAME" \
  -te "$TEST_DATA" \
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
echo "  - SAE model: ${CHECKPOINT}"
echo "  - Training curves: ${LATEST_EXP}/training_curves.png"
echo "  - Top features: ${LATEST_EXP}/analysis/top_features.json"
echo "  - Feature correlations: ${LATEST_EXP}/analysis/feature_correlations.png"
echo ""
echo "Next steps:"
echo "  1. Review training curves: open ${LATEST_EXP}/training_curves.png"
echo "  2. Check top features: cat ${LATEST_EXP}/analysis/top_features.json | head -50"
echo "  3. Review spatial features: cat ${LATEST_EXP}/analysis/dimension_features.json"
echo "  4. Check probe R²: cat ${LATEST_EXP}/analysis/probe_results.json"
echo ""
echo "======================================================"


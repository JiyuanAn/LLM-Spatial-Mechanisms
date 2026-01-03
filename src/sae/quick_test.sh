#!/bin/bash

# 快速测试不同 L1 系数（只训练 3 个 epoch，少量数据）
# 用于快速找到合适的超参数

set -e

echo "======================================================"
echo "Quick L1 Coefficient Test"
echo "======================================================"
echo ""

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="../dataGenerate/spatial_reasoning_dataset_ZH.json"
TEST_DATA="../dataGenerate/spatial_reasoning_dataset_ZH_test.json"
OUTPUT_DIR="./sae_quick_test"
LAYER=8
N_FEATURES=2048
BATCH_SIZE=128
LR=3e-4
NUM_EPOCHS=3
MAX_SAMPLES=1000  # 只用 1000 样本快速测试

# 测试不同的 L1 系数
L1_COEFFS=(5e-5 1e-4 3e-4 5e-4)

echo "Testing L1 coefficients: ${L1_COEFFS[@]}"
echo "Using only $MAX_SAMPLES samples for quick test"
echo ""

for l1 in "${L1_COEFFS[@]}"; do
    echo ""
    echo "======================================================"
    echo "Testing L1=$l1"
    echo "======================================================"
    
    python train_sae.py \
      -m "$MODEL_NAME" \
      -tr "$TRAIN_DATA" \
      -te "$TEST_DATA" \
      --layer $LAYER \
      --n_features $N_FEATURES \
      --l1_coeff $l1 \
      --batch_size $BATCH_SIZE \
      --lr $LR \
      --num_epochs $NUM_EPOCHS \
      --max_samples $MAX_SAMPLES \
      --output_dir "$OUTPUT_DIR"
    
    # 找到最新实验
    LATEST_EXP=$(ls -td ${OUTPUT_DIR}/L${LAYER}_F${N_FEATURES}_* 2>/dev/null | head -n1)
    HISTORY_FILE="${LATEST_EXP}/training_history.json"
    
    # 提取关键指标
    if [ -f "$HISTORY_FILE" ]; then
        echo ""
        echo "Results for L1=$l1:"
        python3 << EOF
import json
with open("$HISTORY_FILE", 'r') as f:
    h = json.load(f)
print(f"  Final L0: {h['test_l0'][-1]:.1f}")
print(f"  Final CosSim: {h['test_cos_sim'][-1]:.4f}")
print(f"  Final MSE: {h['test_mse'][-1]:.6f}")
print(f"  L0 stability: {h['test_l0']}")
EOF
    fi
done

echo ""
echo "======================================================"
echo "QUICK TEST COMPLETED"
echo "======================================================"
echo ""
echo "Review results in: $OUTPUT_DIR"
echo ""
echo "Choose the L1 coefficient with:"
echo "  - L0 in range 20-200"
echo "  - L0 stable across epochs (not collapsing to 0)"
echo "  - CosSim > 0.90"
echo ""
echo "Then update run_experiment.sh with the best L1 value"
echo "======================================================"




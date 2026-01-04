#!/bin/bash

# SAELens 多层实验一键运行脚本
# 对模型的每一层执行训练和分析

set -e

# 使用 SA conda 环境的 Python
PYTHON_BIN="$HOME/.conda/envs/SA/bin/python"

echo "======================================================"
echo "SAELens Multi-Layer Training and Analysis Pipeline"
echo "======================================================"
echo ""

# 配置参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/spatial_reasoning_dataset_EN_with_prompt.json"
TEST_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/spatial_reasoning_dataset_EN_test_with_prompt.json"
OUTPUT_DIR="./sae_results_all_layers_task_1"
N_FEATURES=4096
L1_COEFF=1e-5
LR=3e-4
NUM_TOKENS=100000
TOP_K=50

# 层范围配置 (Qwen2.5-7B 有 28 层，编号 0-27)
START_LAYER=0
END_LAYER=27

# 可选：限制样本数量（快速测试）
# MAX_SAMPLES="--max_samples 5000"
MAX_SAMPLES=""

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Layer Range: $START_LAYER to $END_LAYER"
echo "  Features: $N_FEATURES"
echo "  L1 Coefficient: $L1_COEFF"
echo "  Training Tokens: $NUM_TOKENS"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# 创建日志目录
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# 记录成功和失败的层
SUCCESS_LAYERS=()
FAILED_LAYERS=()

# 开始时间
START_TIME=$(date +%s)

# ====================================================
# 主循环：遍历每一层
# ====================================================
for LAYER in $(seq $START_LAYER $END_LAYER); do
    echo "======================================================"
    echo "Processing Layer $LAYER ($(($LAYER - $START_LAYER + 1))/$(($END_LAYER - $START_LAYER + 1)))"
    echo "======================================================"
    echo ""
    
    LAYER_START_TIME=$(date +%s)
    LOG_FILE="${LOG_DIR}/layer_${LAYER}.log"
    
    # 尝试训练和分析当前层
    if (
        set -e
        
        # ====================================================
        # Step 1: 训练 SAE
        # ====================================================
        echo "Step 1: Training SAE for Layer $LAYER" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
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
          $MAX_SAMPLES 2>&1 | tee -a "$LOG_FILE"
        
        echo "" | tee -a "$LOG_FILE"
        echo "✓ SAE training completed for Layer $LAYER!" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        # ====================================================
        # Step 2: 找到最新的 checkpoint
        # ====================================================
        echo "Step 2: Locating checkpoint for Layer $LAYER" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        LATEST_EXP=$(ls -td ${OUTPUT_DIR}/L${LAYER}_F${N_FEATURES}_* 2>/dev/null | head -n1)
        
        if [ -z "$LATEST_EXP" ]; then
            echo "Error: No checkpoint found for Layer $LAYER" | tee -a "$LOG_FILE"
            exit 1
        fi
        
        echo "Found checkpoint: $LATEST_EXP" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        # ====================================================
        # Step 3: 分析特征
        # ====================================================
        echo "Step 3: Analyzing Features for Layer $LAYER" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        $PYTHON_BIN analyze_features_saelens_new.py \
          -c "$LATEST_EXP" \
          --top_k $TOP_K 2>&1 | tee -a "$LOG_FILE"
        
        echo "" | tee -a "$LOG_FILE"
        echo "✓ Feature analysis completed for Layer $LAYER!" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
    ); then
        # 成功
        SUCCESS_LAYERS+=($LAYER)
        LAYER_END_TIME=$(date +%s)
        LAYER_DURATION=$((LAYER_END_TIME - LAYER_START_TIME))
        echo "✓ Layer $LAYER completed successfully in ${LAYER_DURATION}s" | tee -a "$LOG_FILE"
        echo ""
    else
        # 失败
        FAILED_LAYERS+=($LAYER)
        echo "✗ Layer $LAYER failed! Check log: $LOG_FILE" | tee -a "$LOG_FILE"
        echo "Continuing with next layer..."
        echo ""
    fi
    
    echo ""
done

# ====================================================
# 总结
# ====================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "======================================================"
echo "ALL LAYERS PROCESSING COMPLETED"
echo "======================================================"
echo ""
echo "Total Time: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)"
echo ""
echo "Successfully processed layers (${#SUCCESS_LAYERS[@]}):"
if [ ${#SUCCESS_LAYERS[@]} -gt 0 ]; then
    echo "  ${SUCCESS_LAYERS[@]}"
else
    echo "  None"
fi
echo ""

if [ ${#FAILED_LAYERS[@]} -gt 0 ]; then
    echo "Failed layers (${#FAILED_LAYERS[@]}):"
    echo "  ${FAILED_LAYERS[@]}"
    echo ""
    echo "Check logs for details:"
    for LAYER in "${FAILED_LAYERS[@]}"; do
        echo "  - ${LOG_DIR}/layer_${LAYER}.log"
    done
    echo ""
fi

echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Next steps:"
echo "  1. Review all training results: ls -lh ${OUTPUT_DIR}/"
echo "  2. Compare layer features: check ${OUTPUT_DIR}/L*_F*/analysis/"
echo "  3. Check logs: ls -lh ${LOG_DIR}/"
echo ""
echo "======================================================"


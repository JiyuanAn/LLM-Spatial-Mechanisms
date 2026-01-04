#!/bin/bash
# SAE Feature Ablation Experiment Runner for All Layers
# 对所有层运行 SAE 特征干预实验

set -e  # Exit on error

echo "======================================================"
echo "SAE Feature Ablation Experiment - All Layers"
echo "======================================================"
echo ""

# 默认参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
EVAL_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/spatial_reasoning_dataset_EN_test_with_prompt.json"
MAX_SAMPLES=100  # 可以调整，全量测试设为 None
VERSION="simple"  # 使用简化版本（更快更稳定）

# SAE 结果目录（包含所有层的 checkpoint）
SAE_RESULTS_DIR="./sae_results_all_layers_task_1"
N_FEATURES=4096

# 层范围配置 (Qwen2.5-7B 有 28 层，编号 0-27)
START_LAYER=0
END_LAYER=27

# 从命令行获取参数
if [ ! -z "$1" ]; then
    SAE_RESULTS_DIR=$1
fi

if [ ! -z "$2" ]; then
    MAX_SAMPLES=$2
fi

if [ ! -z "$3" ]; then
    VERSION=$3
fi

# 检查 SAE 结果目录是否存在
if [ ! -d "$SAE_RESULTS_DIR" ]; then
    echo "Error: SAE results directory not found: $SAE_RESULTS_DIR"
    echo ""
    echo "Usage: bash run_ablation_all_layers.sh [sae_results_dir] [max_samples] [version]"
    echo ""
    echo "Arguments:"
    echo "  sae_results_dir: Directory containing SAE checkpoints for all layers"
    echo "                   (default: ./sae_results_all_layers_task_1)"
    echo "  max_samples: Maximum number of samples to evaluate (default: 100)"
    echo "  version: 'simple' (logits-based, fast) or 'full' (generation-based)"
    echo ""
    echo "Example:"
    echo "  bash run_ablation_all_layers.sh ./sae_results_all_layers_task_1 100 simple"
    echo ""
    exit 1
fi

# 选择脚本
if [ "$VERSION" == "simple" ]; then
    SCRIPT="ablate_features_simple_new.py"
    echo "Using simplified version (logits-based, faster)"
else
    SCRIPT="ablate_features_saelens_new.py"
    echo "Using full version (generation-based)"
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  SAE Results Directory: $SAE_RESULTS_DIR"
echo "  Layer Range: $START_LAYER to $END_LAYER"
echo "  Eval Data: $EVAL_DATA"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Version: $VERSION"
echo ""

# 检查评估数据是否存在
if [ ! -f "$EVAL_DATA" ]; then
    echo "Error: Eval data not found: $EVAL_DATA"
    echo "Please modify EVAL_DATA in this script"
    exit 1
fi

# 创建日志目录
LOG_DIR="${SAE_RESULTS_DIR}/ablation_logs"
mkdir -p "$LOG_DIR"

# 记录成功和失败的层
SUCCESS_LAYERS=()
FAILED_LAYERS=()
SKIPPED_LAYERS=()

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
    LOG_FILE="${LOG_DIR}/ablation_layer_${LAYER}.log"
    
    # 查找该层的 SAE checkpoint
    SAE_CHECKPOINT=$(ls -td ${SAE_RESULTS_DIR}/L${LAYER}_F${N_FEATURES}_* 2>/dev/null | head -n1)
    
    if [ -z "$SAE_CHECKPOINT" ]; then
        echo "⚠ Warning: No SAE checkpoint found for Layer $LAYER" | tee -a "$LOG_FILE"
        echo "  Expected pattern: ${SAE_RESULTS_DIR}/L${LAYER}_F${N_FEATURES}_*" | tee -a "$LOG_FILE"
        SKIPPED_LAYERS+=($LAYER)
        echo ""
        continue
    fi
    
    echo "Found checkpoint: $SAE_CHECKPOINT" | tee -a "$LOG_FILE"
    
    # 检查 analysis 目录是否存在
    ANALYSIS_DIR="${SAE_CHECKPOINT}/analysis"
    if [ ! -d "$ANALYSIS_DIR" ]; then
        echo "⚠ Warning: Analysis directory not found: $ANALYSIS_DIR" | tee -a "$LOG_FILE"
        echo "  Skipping Layer $LAYER - please run analyze_features first" | tee -a "$LOG_FILE"
        SKIPPED_LAYERS+=($LAYER)
        echo ""
        continue
    fi
    
    echo "" | tee -a "$LOG_FILE"
    
    # 尝试运行消融实验
    if (
        set -e
        
        echo "Running ablation experiment for Layer $LAYER..." | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        # 运行实验
        if [ "$MAX_SAMPLES" == "None" ] || [ "$MAX_SAMPLES" == "none" ]; then
            python "$SCRIPT" \
                --model_name "$MODEL_NAME" \
                --sae_checkpoint "$SAE_CHECKPOINT" \
                --eval_data_file "$EVAL_DATA" 2>&1 | tee -a "$LOG_FILE"
        else
            python "$SCRIPT" \
                --model_name "$MODEL_NAME" \
                --sae_checkpoint "$SAE_CHECKPOINT" \
                --eval_data_file "$EVAL_DATA" \
                --max_samples "$MAX_SAMPLES" 2>&1 | tee -a "$LOG_FILE"
        fi
        
        echo "" | tee -a "$LOG_FILE"
        
    ); then
        # 成功
        SUCCESS_LAYERS+=($LAYER)
        LAYER_END_TIME=$(date +%s)
        LAYER_DURATION=$((LAYER_END_TIME - LAYER_START_TIME))
        echo "✓ Layer $LAYER ablation completed successfully in ${LAYER_DURATION}s" | tee -a "$LOG_FILE"
        echo "  Results: ${SAE_CHECKPOINT}/ablation_results/" | tee -a "$LOG_FILE"
        echo ""
    else
        # 失败
        FAILED_LAYERS+=($LAYER)
        echo "✗ Layer $LAYER ablation failed! Check log: $LOG_FILE"
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

SUMMARY_FILE="${SAE_RESULTS_DIR}/ablation_all_layers_summary.txt"

{
    echo "======================================================"
    echo "ALL LAYERS ABLATION COMPLETED"
    echo "======================================================"
    echo ""
    echo "Timestamp: $(date)"
    echo "Total Time: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)"
    echo ""
    echo "Configuration:"
    echo "  Model: $MODEL_NAME"
    echo "  SAE Results Directory: $SAE_RESULTS_DIR"
    echo "  Features: $N_FEATURES"
    echo "  Max Samples: $MAX_SAMPLES"
    echo "  Version: $VERSION"
    echo ""
    echo "Successfully processed layers (${#SUCCESS_LAYERS[@]}):"
    if [ ${#SUCCESS_LAYERS[@]} -gt 0 ]; then
        echo "  ${SUCCESS_LAYERS[@]}"
    else
        echo "  None"
    fi
    echo ""
    
    if [ ${#SKIPPED_LAYERS[@]} -gt 0 ]; then
        echo "Skipped layers (${#SKIPPED_LAYERS[@]}):"
        echo "  ${SKIPPED_LAYERS[@]}"
        echo "  (No checkpoint or analysis found)"
        echo ""
    fi
    
    if [ ${#FAILED_LAYERS[@]} -gt 0 ]; then
        echo "Failed layers (${#FAILED_LAYERS[@]}):"
        echo "  ${FAILED_LAYERS[@]}"
        echo ""
        echo "Check logs for details:"
        for LAYER in "${FAILED_LAYERS[@]}"; do
            echo "  - ${LOG_DIR}/ablation_layer_${LAYER}.log"
        done
        echo ""
    fi
    
    echo "Results locations:"
    for LAYER in "${SUCCESS_LAYERS[@]}"; do
        CHECKPOINT=$(ls -td ${SAE_RESULTS_DIR}/L${LAYER}_F${N_FEATURES}_* 2>/dev/null | head -n1)
        if [ ! -z "$CHECKPOINT" ]; then
            echo "  Layer $LAYER: ${CHECKPOINT}/ablation_results/"
        fi
    done
    echo ""
    
    echo "Logs saved to: $LOG_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Review summary: cat $SUMMARY_FILE"
    echo "  2. Compare ablation results across layers"
    echo "  3. Check individual layer results in ablation_results/ directories"
    echo "  4. Analyze which layers are most important for spatial reasoning"
    echo ""
    echo "======================================================"
} | tee "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo ""


#!/bin/bash
# 运行 Gradient-Based Feature Attribution 分析

set -e  # Exit on error

# =========================
# 配置
# =========================
SAE_CHECKPOINT="${1:-./sae_results_new/L18_F16384_L11e-05_20260102_232048/}"
MAX_SAMPLES="${2:-100}"
METHOD="${3:-grad_x_act}"  # grad_x_act, grad_norm, integrated_gradients

# 数据路径
EVAL_DATA="../data_generation/spatial_reasoning_dataset_EN_test_with_prompt.json"

# 模型
MODEL="Qwen/Qwen2.5-7B-Instruct"

# =========================
# 打印配置
# =========================
echo "======================================"
echo "Gradient-Based Feature Attribution"
echo "======================================"
echo ""
echo "Configuration:"
echo "  SAE Checkpoint: $SAE_CHECKPOINT"
echo "  Eval Data: $EVAL_DATA"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Attribution Method: $METHOD"
echo "  Model: $MODEL"
echo ""

# 检查 checkpoint 是否存在
if [ ! -d "$SAE_CHECKPOINT" ]; then
    echo "Error: SAE checkpoint directory not found: $SAE_CHECKPOINT"
    echo ""
    echo "Usage: bash run_gradient_attribution.sh [SAE_CHECKPOINT] [MAX_SAMPLES] [METHOD]"
    echo ""
    echo "Example:"
    echo "  bash run_gradient_attribution.sh ./sae_results/L18_F16384_L11e-05_20260102_232048/ 100 grad_x_act"
    echo ""
    exit 1
fi

# 检查数据文件是否存在
if [ ! -f "$EVAL_DATA" ]; then
    echo "Error: Eval data file not found: $EVAL_DATA"
    echo "Please update EVAL_DATA path in the script."
    exit 1
fi

# =========================
# 运行 Gradient Attribution
# =========================
echo "======================================"
echo "Step 1: Computing Gradient Attribution"
echo "======================================"
echo ""

python gradient_attribution.py \
    -m "$MODEL" \
    -c "$SAE_CHECKPOINT" \
    -e "$EVAL_DATA" \
    --max_samples $MAX_SAMPLES \
    --attribution_method $METHOD \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Gradient attribution failed!"
    exit 1
fi

echo ""
echo "✓ Gradient attribution completed"

# =========================
# 可视化结果
# =========================
echo ""
echo "======================================"
echo "Step 2: Visualizing Results"
echo "======================================"
echo ""

# 找到最新的结果文件
RESULT_FILE=$(ls -t ${SAE_CHECKPOINT}/gradient_attribution/gradient_attribution_${METHOD}_*.json | head -1)

if [ -z "$RESULT_FILE" ]; then
    echo "Warning: No result file found, skipping visualization"
else
    echo "Visualizing: $RESULT_FILE"
    
    python visualize_gradient_attribution.py \
        -r "$RESULT_FILE" \
        -c "$SAE_CHECKPOINT"
    
    if [ $? -eq 0 ]; then
        echo "✓ Visualization completed"
    else
        echo "Warning: Visualization failed, but continuing..."
    fi
fi

# =========================
# 完成
# =========================
echo ""
echo "======================================"
echo "Gradient Attribution Complete!"
echo "======================================"
echo ""
echo "Results saved to: ${SAE_CHECKPOINT}/gradient_attribution/"
echo ""
echo "Next steps:"
echo "  1. View the summary: cat ${SAE_CHECKPOINT}/gradient_attribution/gradient_summary_${METHOD}_*.txt"
echo "  2. Check visualizations in gradient_attribution/ folder"
echo "  3. Compare with ablation results using compare_methods.py"
echo ""
echo "To try other attribution methods:"
echo "  - grad_x_act (default): Gradient × Activation"
echo "  - grad_norm: Gradient norm"
echo "  - integrated_gradients: Integrated Gradients (slower but more accurate)"
echo ""
echo "Example:"
echo "  bash run_gradient_attribution.sh $SAE_CHECKPOINT $MAX_SAMPLES integrated_gradients"
echo ""




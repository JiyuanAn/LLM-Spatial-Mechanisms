#!/bin/bash
# 快速分析单个特征

if [ $# -eq 0 ]; then
    echo "Usage: bash quick_analyze.sh <feature_id>"
    echo ""
    echo "Example: bash quick_analyze.sh 1285"
    echo ""
    echo "Recommended features to analyze:"
    echo "  - 1285 (strongest, mean=2.37)"
    echo "  - 905  (2nd strongest, mean=2.31)"
    echo "  - 2571 (3rd strongest, mean=1.71)"
    echo "  - 1612 (4th strongest, mean=1.68)"
    echo "  - 694  (5th strongest, mean=1.66)"
    exit 1
fi

FEATURE_ID=$1
CHECKPOINT="./sae_experiment_20260102_005950/sae_step_5000.pt"
MODEL_PATH="/home/s202507009/workspace/LLMs/Qwen2.5/Qwen2.5-7B-Instruct"
DATA_PATH="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/dataGenerate/spatial_reasoning_dataset_EN.json"

echo "=========================================="
echo "Analyzing Feature $FEATURE_ID"
echo "=========================================="
echo ""

# 分析特征
python analyze_features.py \
    --checkpoint "$CHECKPOINT" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --feature_id "$FEATURE_ID" \
    --top_k 30 \
    --output "feature_${FEATURE_ID}_examples.json" \
    --batch_size 8

echo ""
echo "=========================================="
echo "Visualizing Feature $FEATURE_ID"
echo "=========================================="
echo ""

# 可视化
python visualize_features.py \
    --feature "feature_${FEATURE_ID}_examples.json" \
    --output "feature_${FEATURE_ID}_report.txt"

echo ""
echo "=========================================="
echo "✅ Analysis Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - JSON: feature_${FEATURE_ID}_examples.json"
echo "  - Report: feature_${FEATURE_ID}_report.txt"
echo ""
echo "View report:"
echo "  cat feature_${FEATURE_ID}_report.txt"
echo "  less feature_${FEATURE_ID}_report.txt"



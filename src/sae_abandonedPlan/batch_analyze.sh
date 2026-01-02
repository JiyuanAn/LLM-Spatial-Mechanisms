#!/bin/bash
# 批量分析多个 SAE 特征

CHECKPOINT="./sae_experiment_20260102_005950/sae_step_5000.pt"
MODEL_PATH="/home/s202507009/workspace/LLMs/Qwen2.5/Qwen2.5-7B-Instruct"
DATA_PATH="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/dataGenerate/spatial_reasoning_dataset_EN.json"
OUTPUT_DIR="./feature_analysis_results"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Step 1: Analyzing all features statistics"
echo "=========================================="

python analyze_features.py \
    --checkpoint "$CHECKPOINT" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --analyze_all \
    --output "$OUTPUT_DIR/all_features_stats.json" \
    --batch_size 8

echo ""
echo "=========================================="
echo "Step 2: Analyzing specific features"
echo "=========================================="

# 分析前 10 个特征（可以根据统计结果修改）
for FEATURE_ID in 0 1 2 3 4 5 10 20 50 100; do
    echo "Analyzing feature $FEATURE_ID..."
    python analyze_features.py \
        --checkpoint "$CHECKPOINT" \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --feature_id "$FEATURE_ID" \
        --top_k 20 \
        --output "$OUTPUT_DIR/feature_${FEATURE_ID}_top_examples.json"
done

echo ""
echo "=========================================="
echo "Analysis completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="


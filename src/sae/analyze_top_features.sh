#!/bin/bash
# 分析 SAE 中最重要的特征

CHECKPOINT="./sae_experiment_20260102_005950/sae_step_5000.pt"
MODEL_PATH="/home/s202507009/workspace/LLMs/Qwen2.5/Qwen2.5-7B-Instruct"
DATA_PATH="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/dataGenerate/spatial_reasoning_dataset_EN.json"
OUTPUT_DIR="./top_features_analysis"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Analyzing Top SAE Features"
echo "=========================================="
echo ""

# Top 5 最强激活特征
TOP_FEATURES=(1285 905 2571 1612 694)

echo "🔥 Analyzing Top 5 strongest features..."
for FID in "${TOP_FEATURES[@]}"; do
    echo "  → Feature $FID"
    python analyze_features.py \
        --checkpoint "$CHECKPOINT" \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --feature_id "$FID" \
        --top_k 30 \
        --output "$OUTPUT_DIR/feature_${FID}_examples.json" \
        --batch_size 8
    
    echo "  → Visualizing Feature $FID"
    python visualize_features.py \
        --feature "$OUTPUT_DIR/feature_${FID}_examples.json" \
        --output "$OUTPUT_DIR/feature_${FID}_report.txt"
    
    echo ""
done

echo "=========================================="
echo "🎯 Additional interesting features"
echo "=========================================="

# 其他有趣的特征（高频率）
ADDITIONAL_FEATURES=(251 542 377 1070 264 332 547 694 718)

for FID in "${ADDITIONAL_FEATURES[@]}"; do
    echo "  → Feature $FID"
    python analyze_features.py \
        --checkpoint "$CHECKPOINT" \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --feature_id "$FID" \
        --top_k 30 \
        --output "$OUTPUT_DIR/feature_${FID}_examples.json" \
        --batch_size 8
    
    python visualize_features.py \
        --feature "$OUTPUT_DIR/feature_${FID}_examples.json" \
        --output "$OUTPUT_DIR/feature_${FID}_report.txt"
    
    echo ""
done

echo "=========================================="
echo "✅ Analysis Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Quick view of top features:"
echo ""

for FID in "${TOP_FEATURES[@]}"; do
    echo "===== Feature $FID ====="
    head -30 "$OUTPUT_DIR/feature_${FID}_report.txt"
    echo ""
done

echo "=========================================="
echo "📝 Summary of all analyzed features:"
ls -lh "$OUTPUT_DIR"/*.txt
echo "=========================================="



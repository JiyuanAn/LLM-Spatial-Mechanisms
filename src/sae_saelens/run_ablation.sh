#!/bin/bash
# SAE Feature Ablation Experiment Runner
# 运行 SAE 特征干预实验

set -e  # Exit on error

echo "======================================================"
echo "SAE Feature Ablation Experiment"
echo "======================================================"
echo ""

# 默认参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
EVAL_DATA="../../data/spatial/test_data.json"  # 修改为你的测试数据路径
MAX_SAMPLES=100  # 可以调整，全量测试设为 None
VERSION="simple"  # 使用简化版本（更快更稳定）

# 从命令行获取 SAE checkpoint 路径
if [ -z "$1" ]; then
    echo "Usage: bash run_ablation.sh <sae_checkpoint_dir> [max_samples] [version]"
    echo ""
    echo "Arguments:"
    echo "  sae_checkpoint_dir: Path to SAE checkpoint directory"
    echo "  max_samples: Maximum number of samples to evaluate (default: 100)"
    echo "  version: 'simple' (logits-based, fast) or 'full' (generation-based)"
    echo ""
    echo "Example:"
    echo "  bash run_ablation.sh ./sae_results/L8_F2048_L10.0001_20260102_120000 100 simple"
    echo ""
    exit 1
fi

SAE_CHECKPOINT=$1

if [ ! -d "$SAE_CHECKPOINT" ]; then
    echo "Error: SAE checkpoint directory not found: $SAE_CHECKPOINT"
    exit 1
fi

# 检查是否已经完成 analysis
ANALYSIS_DIR="${SAE_CHECKPOINT}/analysis"
if [ ! -d "$ANALYSIS_DIR" ]; then
    echo "Error: Analysis directory not found: $ANALYSIS_DIR"
    echo "Please run analyze_features_saelens.py first:"
    echo "  python analyze_features_saelens.py -c $SAE_CHECKPOINT"
    exit 1
fi

# 可选：从命令行指定 max_samples
if [ ! -z "$2" ]; then
    MAX_SAMPLES=$2
fi

# 可选：从命令行指定版本
if [ ! -z "$3" ]; then
    VERSION=$3
fi

# 选择脚本
if [ "$VERSION" == "simple" ]; then
    SCRIPT="ablate_features_simple.py"
    echo "Using simplified version (logits-based, faster)"
else
    SCRIPT="ablate_features_saelens.py"
    echo "Using full version (generation-based)"
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  SAE Checkpoint: $SAE_CHECKPOINT"
echo "  Eval Data: $EVAL_DATA"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Version: $VERSION"
echo ""

# 检查评估数据是否存在
if [ ! -f "$EVAL_DATA" ]; then
    echo "Warning: Eval data not found: $EVAL_DATA"
    echo "Please modify EVAL_DATA in this script or provide the correct path"
    read -p "Enter eval data path: " EVAL_DATA
fi

echo "======================================================"
echo "Running ablation experiment..."
echo "======================================================"
echo ""

# 运行实验
if [ "$MAX_SAMPLES" == "None" ] || [ "$MAX_SAMPLES" == "none" ]; then
    python "$SCRIPT" \
        --model_name "$MODEL_NAME" \
        --sae_checkpoint "$SAE_CHECKPOINT" \
        --eval_data_file "$EVAL_DATA"
else
    python "$SCRIPT" \
        --model_name "$MODEL_NAME" \
        --sae_checkpoint "$SAE_CHECKPOINT" \
        --eval_data_file "$EVAL_DATA" \
        --max_samples "$MAX_SAMPLES"
fi

echo ""
echo "======================================================"
echo "Ablation experiment complete!"
echo "======================================================"
echo ""
echo "Results saved to: ${SAE_CHECKPOINT}/ablation_results/"
echo ""
echo "Next steps:"
echo "  1. Check ablation_summary_*.txt for quick results"
echo "  2. Check ablation_results_*.json for detailed data"
echo "  3. Analyze examples to understand failure modes"
echo ""


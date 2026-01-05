#!/bin/bash
# =============================================================================
# 快速修复脚本 - 解决特征死亡问题
# Quick Fix Script for Feature Death Problem
# =============================================================================

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     SAE 特征死亡问题 - 快速修复脚本                    ║${NC}"
echo -e "${BLUE}║     Quick Fix for Feature Death Problem               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# 数据路径
TRAIN_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_with_prompt.json"
TEST_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test_with_prompt.json"
LAYER=18
OUTPUT_DIR="./sae_results_new"

# =============================================================================
# 显示当前问题
# =============================================================================
echo -e "${RED}当前问题诊断：${NC}"
echo "  ❌ Active features: 161 / 16384 (0.98%)"
echo "  ❌ Dead features: 16192 (98.8%)"
echo "  ❌ Probe R²: -0.0083 (负值！)"
echo "  ❌ Spatial features: 仅 2 个"
echo ""
echo -e "${YELLOW}根本原因：${NC}"
echo "  1. L1 coefficient 太高 (1e-5) → 过度稀疏化"
echo "  2. 特征数量太多 (16384) vs 数据量 (2000 samples)"
echo "  3. 只使用最后一个 token → 数据利用率低"
echo ""

# =============================================================================
# 提供多个方案
# =============================================================================
echo -e "${BLUE}可选修复方案：${NC}"
echo ""
echo "方案 1 (推荐): 使用改进的训练脚本 + 优化参数"
echo "  • 特征数: 8192 (减半)"
echo "  • L1 系数: 1e-6 (降低 10倍)"
echo "  • 使用所有 token 位置 (增加数据量)"
echo "  • 更大的 batch size (128)"
echo ""
echo "方案 2 (保守): 仅降低 L1 系数"
echo "  • 特征数: 16384 (保持)"
echo "  • L1 系数: 1e-6 (降低 10倍)"
echo "  • 其他保持原样"
echo ""
echo "方案 3 (激进): 小字典 + 低 L1"
echo "  • 特征数: 4096 (减少 75%)"
echo "  • L1 系数: 5e-7 (降低 20倍)"
echo "  • 使用所有 token"
echo ""

# =============================================================================
# 用户选择
# =============================================================================
echo -e "${GREEN}请选择要运行的方案 (1/2/3):${NC}"
read -p "输入选项 [1]: " CHOICE
CHOICE=${CHOICE:-1}

case $CHOICE in
    1)
        echo -e "${GREEN}✓ 选择方案 1: 改进的训练 (推荐)${NC}"
        SCRIPT="train_sae_saelens_improved.py"
        N_FEATURES=8192
        L1_COEFF=1e-6
        EXTRA_ARGS="--use_all_tokens --batch_size 128 --n_epochs 15"
        ;;
    2)
        echo -e "${GREEN}✓ 选择方案 2: 保守修复${NC}"
        SCRIPT="train_sae_saelens.py"
        N_FEATURES=16384
        L1_COEFF=1e-6
        EXTRA_ARGS=""
        ;;
    3)
        echo -e "${GREEN}✓ 选择方案 3: 激进修复${NC}"
        SCRIPT="train_sae_saelens_improved.py"
        N_FEATURES=4096
        L1_COEFF=5e-7
        EXTRA_ARGS="--use_all_tokens --batch_size 128 --n_epochs 15"
        ;;
    *)
        echo -e "${RED}无效选项，使用默认方案 1${NC}"
        SCRIPT="train_sae_saelens_improved.py"
        N_FEATURES=8192
        L1_COEFF=1e-6
        EXTRA_ARGS="--use_all_tokens --batch_size 128 --n_epochs 15"
        ;;
esac

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}开始训练${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "配置:"
echo "  Script: $SCRIPT"
echo "  Features: $N_FEATURES"
echo "  L1: $L1_COEFF"
echo "  Layer: $LAYER"
echo "  Extra: $EXTRA_ARGS"
echo ""

# 确认
read -p "按 Enter 继续，或 Ctrl+C 取消..."

# =============================================================================
# 运行训练
# =============================================================================
echo ""
echo -e "${GREEN}>>> Step 1: 训练 SAE${NC}"
echo ""

python $SCRIPT \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data_file_path "$TRAIN_DATA" \
    --test_data_file_path "$TEST_DATA" \
    --layer $LAYER \
    --n_features $N_FEATURES \
    --l1_coeff $L1_COEFF \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 训练失败${NC}"
    exit 1
fi

# 找到最新的 checkpoint
CHECKPOINT=$(ls -td ${OUTPUT_DIR}/L${LAYER}_F${N_FEATURES}_* 2>/dev/null | head -n1)

if [ -z "$CHECKPOINT" ]; then
    echo -e "${RED}✗ 找不到 checkpoint${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ 训练完成！${NC}"
echo "  Checkpoint: $CHECKPOINT"
echo ""

# =============================================================================
# 运行分析
# =============================================================================
echo -e "${GREEN}>>> Step 2: 分析特征${NC}"
echo ""

python analyze_features_saelens.py \
    --checkpoint "$CHECKPOINT" \
    --top_k 50

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 分析失败${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ 分析完成！${NC}"
echo ""

# =============================================================================
# 显示结果对比
# =============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}结果对比${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# 提取关键指标
TRAIN_HISTORY="$CHECKPOINT/training_history.json"
CONFIG="$CHECKPOINT/config.json"

if [ -f "$TRAIN_HISTORY" ]; then
    # 使用 Python 提取最终 L0
    FINAL_L0=$(python3 -c "import json; data=json.load(open('$TRAIN_HISTORY')); print(f\"{data['test_l0'][-1]:.1f}\")")
    TOTAL_FEATURES=$(python3 -c "import json; data=json.load(open('$CONFIG')); print(data['n_features'])")
    ACTIVE_PCT=$(python3 -c "import json; hist=json.load(open('$TRAIN_HISTORY')); cfg=json.load(open('$CONFIG')); print(f\"{100*hist['test_l0'][-1]/cfg['n_features']:.1f}\")")
    
    echo "训练结果:"
    echo "  Total features: $TOTAL_FEATURES"
    echo "  Active features: $FINAL_L0 ($ACTIVE_PCT%)"
    
    # 对比
    if (( $(echo "$ACTIVE_PCT < 1.0" | bc -l) )); then
        echo -e "  ${RED}❌ 仍然太稀疏！建议进一步降低 L1${NC}"
        echo -e "     尝试: L1 = $(python3 -c "print(f'{$L1_COEFF/2:.1e}')")"
    elif (( $(echo "$ACTIVE_PCT < 5.0" | bc -l) )); then
        echo -e "  ${YELLOW}⚠️  还是比较稀疏，可以继续降低 L1${NC}"
    elif (( $(echo "$ACTIVE_PCT > 30.0" | bc -l) )); then
        echo -e "  ${YELLOW}⚠️  不够稀疏，建议提高 L1${NC}"
    else
        echo -e "  ${GREEN}✓ 稀疏性看起来健康！${NC}"
    fi
fi

echo ""

# 检查 analysis 结果
ANALYSIS_DIR="$CHECKPOINT/analysis"
if [ -f "$ANALYSIS_DIR/dimension_features.json" ]; then
    echo "空间特征统计:"
    
    # 提取各个方向的特征数量
    python3 << EOF
import json
with open('$ANALYSIS_DIR/dimension_features.json') as f:
    dim_features = json.load(f)

labels = {
    'X_positive': 'Right',
    'X_negative': 'Left',
    'Y_positive': 'Above',
    'Y_negative': 'Below',
    'Z_positive': 'Front',
    'Z_negative': 'Behind',
}

total = 0
for key, label in labels.items():
    count = len(dim_features[key])
    total += count
    print(f"  {label:10s}: {count:3d} features")

print(f"  {'Total':10s}: {total:3d} features")

if total < 5:
    print("  \033[0;31m❌ 空间特征太少！\033[0m")
elif total < 20:
    print("  \033[1;33m⚠️  空间特征偏少\033[0m")
else:
    print("  \033[0;32m✓ 找到足够的空间特征\033[0m")
EOF
fi

echo ""

# Probe 结果
if [ -f "$ANALYSIS_DIR/probe_results.json" ]; then
    R2=$(python3 -c "import json; data=json.load(open('$ANALYSIS_DIR/probe_results.json')); print(f\"{data['r2_full']:.4f}\")")
    echo "Probe 性能:"
    echo "  R² = $R2"
    
    if (( $(echo "$R2 < 0" | bc -l) )); then
        echo -e "  ${RED}❌ 负 R²! 模型没有学到有用信息${NC}"
    elif (( $(echo "$R2 < 0.3" | bc -l) )); then
        echo -e "  ${YELLOW}⚠️  R² 偏低，特征质量不高${NC}"
    elif (( $(echo "$R2 < 0.5" | bc -l) )); then
        echo -e "  ${GREEN}✓ R² 可接受${NC}"
    else:
        echo -e "  ${GREEN}✓ R² 很好！${NC}"
    fi
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}完成！${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "结果目录: $CHECKPOINT"
echo ""
echo "查看详细结果:"
echo "  cat $CHECKPOINT/config.json"
echo "  cat $CHECKPOINT/training_history.json | jq '.test_l0'"
echo "  cat $ANALYSIS_DIR/dimension_features.json | jq"
echo ""
echo "查看可视化:"
echo "  ls $ANALYSIS_DIR/*.png"
echo ""

# 下一步建议
echo -e "${YELLOW}下一步建议:${NC}"
if [ -f "$TRAIN_HISTORY" ]; then
    ACTIVE_PCT=$(python3 -c "import json; hist=json.load(open('$TRAIN_HISTORY')); cfg=json.load(open('$CONFIG')); print(f\"{100*hist['test_l0'][-1]/cfg['n_features']:.1f}\")")
    
    if (( $(echo "$ACTIVE_PCT < 3.0" | bc -l) )); then
        NEW_L1=$(python3 -c "print(f'{$L1_COEFF/2:.1e}')")
        echo "  1. 仍然太稀疏，再次运行并降低 L1:"
        echo "     bash quick_fix.sh  (选择方案 3)"
        echo "     或手动设置: --l1_coeff $NEW_L1"
    elif (( $(echo "$ACTIVE_PCT > 25.0" | bc -l) )); then
        NEW_L1=$(python3 -c "print(f'{$L1_COEFF*1.5:.1e}')")
        echo "  1. 不够稀疏，增加 L1 系数:"
        echo "     --l1_coeff $NEW_L1"
    else
        echo "  1. 稀疏性良好，可以进行 ablation 实验:"
        echo "     bash run_complete_pipeline.sh --checkpoint \"$CHECKPOINT\" --ablation --gradient"
    fi
fi

echo ""
echo "  2. 对比不同配置:"
echo "     运行多个 L1 系数的实验并比较"
echo ""
echo "  3. 尝试不同的层:"
echo "     --layer 12  或  --layer 16"
echo ""


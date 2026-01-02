#!/bin/bash

# SAE 超参数网格搜索
# 用于找到最佳的 L1 coefficient 和 feature 数量组合

set -e

echo "======================================================"
echo "SAE Hyperparameter Grid Search"
echo "======================================================"
echo ""

# 固定参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="../dataGenerate/spatial_reasoning_dataset_ZH.json"
TEST_DATA="../dataGenerate/spatial_reasoning_dataset_ZH_test.json"
OUTPUT_DIR="./sae_results_grid"
LAYER=8
BATCH_SIZE=128
LR=3e-4
NUM_EPOCHS=5

# 快速测试模式（可选）
# MAX_SAMPLES="--max_samples 5000"
MAX_SAMPLES=""

# 搜索空间
L1_COEFFS=(5e-4 1e-3 3e-3)  # 稀疏度从低到高
N_FEATURES_LIST=(1024 2048 4096)  # 特征数量

echo "Search space:"
echo "  L1 coefficients: ${L1_COEFFS[@]}"
echo "  Feature counts: ${N_FEATURES_LIST[@]}"
echo "  Total experiments: $((${#L1_COEFFS[@]} * ${#N_FEATURES_LIST[@]}))"
echo ""
echo "This will take approximately $(( ${#L1_COEFFS[@]} * ${#N_FEATURES_LIST[@]} * 15 )) minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# 创建结果汇总文件
SUMMARY_FILE="${OUTPUT_DIR}/grid_search_summary.txt"
mkdir -p "$OUTPUT_DIR"
echo "Experiment,L1_Coeff,N_Features,Train_Loss,Test_Loss,L0,CosSim,Probe_R2" > "$SUMMARY_FILE"

# 实验计数
exp_count=0
total_exp=$((${#L1_COEFFS[@]} * ${#N_FEATURES_LIST[@]}))

# 网格搜索
for n_features in "${N_FEATURES_LIST[@]}"; do
    for l1_coeff in "${L1_COEFFS[@]}"; do
        exp_count=$((exp_count + 1))
        
        echo ""
        echo "======================================================"
        echo "Experiment $exp_count/$total_exp"
        echo "  N_features: $n_features"
        echo "  L1_coeff: $l1_coeff"
        echo "======================================================"
        echo ""
        
        # 训练 SAE
        python train_sae.py \
          -m "$MODEL_NAME" \
          -tr "$TRAIN_DATA" \
          -te "$TEST_DATA" \
          --layer $LAYER \
          --n_features $n_features \
          --l1_coeff $l1_coeff \
          --batch_size $BATCH_SIZE \
          --lr $LR \
          --num_epochs $NUM_EPOCHS \
          --output_dir "$OUTPUT_DIR" \
          $MAX_SAMPLES
        
        # 找到最新的 checkpoint
        LATEST_EXP=$(ls -td ${OUTPUT_DIR}/L${LAYER}_F${n_features}_* 2>/dev/null | head -n1)
        CHECKPOINT="${LATEST_EXP}/sae_checkpoint.pt"
        
        # 分析特征
        python analyze_features.py \
          -c "$CHECKPOINT" \
          -m "$MODEL_NAME" \
          -te "$TEST_DATA" \
          --top_k 50
        
        # 提取关键指标
        HISTORY_FILE="${LATEST_EXP}/training_history.json"
        PROBE_FILE="${LATEST_EXP}/analysis/probe_results.json"
        
        if [ -f "$HISTORY_FILE" ] && [ -f "$PROBE_FILE" ]; then
            # 使用 Python 提取 JSON 数据
            python3 << EOF
import json
import sys

with open("$HISTORY_FILE", 'r') as f:
    history = json.load(f)

with open("$PROBE_FILE", 'r') as f:
    probe = json.load(f)

train_loss = history['train_loss'][-1]
test_loss = history['test_loss'][-1]
l0 = history['test_l0'][-1]
cos_sim = history['test_cos_sim'][-1]
probe_r2 = probe['r2_full']

exp_name = "${LATEST_EXP}".split('/')[-1]

print(f"{exp_name},{l1_coeff},{n_features},{train_loss:.6f},{test_loss:.6f},{l0:.2f},{cos_sim:.4f},{probe_r2:.4f}")
EOF
        fi >> "$SUMMARY_FILE"
        
        echo ""
        echo "✓ Experiment $exp_count/$total_exp completed"
        echo ""
    done
done

# ====================================================
# 生成最终报告
# ====================================================
echo ""
echo "======================================================"
echo "GRID SEARCH COMPLETED"
echo "======================================================"
echo ""
echo "Results summary:"
cat "$SUMMARY_FILE"
echo ""

# 找出最佳配置（按 Probe R² 排序）
echo "Top 3 configurations by Probe R²:"
tail -n +2 "$SUMMARY_FILE" | sort -t',' -k8 -nr | head -n3 | \
    awk -F',' '{printf "  %s: L1=%.0e, F=%d, R²=%.4f, L0=%.1f\n", $1, $2, $3, $8, $6}'
echo ""

echo "Recommendations:"
echo "  1. Choose config with highest Probe R² AND reasonable L0 (20-100)"
echo "  2. If L0 too high: increase L1_coeff"
echo "  3. If R² too low: try more features or different layer"
echo ""
echo "Full results: $SUMMARY_FILE"
echo "All experiments: $OUTPUT_DIR"
echo ""
echo "======================================================"


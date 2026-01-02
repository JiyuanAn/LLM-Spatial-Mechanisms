#!/bin/bash
# SAE 训练示例脚本
# 根据您的实际情况修改以下参数

# 模型路径 - 可以是 HuggingFace 模型名或本地路径
MODEL_PATH="/home/s202507009/workspace/LLMs/Qwen2.5/Qwen2.5-7B-Instruct"

# 数据路径 - JSONL 格式，每行包含 {"prompt": "..."}
DATA_PATH="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/dataGenerate/spatial_reasoning_dataset_EN.json"

# 输出目录
OUT_DIR="./sae_experiment_$(date +%Y%m%d_%H%M%S)"

# 层数 - 选择要分析的 Transformer 层
LAYER=8

# Token 策略 - last, question_mark, 或 answer_prefix
TOKEN_STRATEGY="question_mark"

# SAE 特征数 - 通常是模型维度的 1-16 倍
N_FEATURES=4096

# L1 正则化系数 - 控制稀疏度，值越大越稀疏 (可选 1e-3 或 2e-3)
L1_COEF=1e-3

# 学习率
LR=2e-4

# 批次大小
BATCH_SIZE=4

# 训练步数 (推荐 2000-3000)
MAX_STEPS=2500

# 评估频率
EVAL_EVERY=500

# 随机种子
SEED=42

echo "========================================"
echo "SAE Training Configuration"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUT_DIR"
echo "Layer: $LAYER"
echo "Features: $N_FEATURES"
echo "Steps: $MAX_STEPS"
echo "========================================"

# 运行训练
python train_sae.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --out_dir "$OUT_DIR" \
    --layer "$LAYER" \
    --token_strategy "$TOKEN_STRATEGY" \
    --n_features "$N_FEATURES" \
    --l1_coef "$L1_COEF" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --eval_every "$EVAL_EVERY" \
    --seed "$SEED"

echo "========================================"
echo "Training completed!"
echo "Results saved to: $OUT_DIR"
echo "========================================"


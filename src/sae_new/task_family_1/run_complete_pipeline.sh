#!/bin/bash
# =============================================================================
# SAE 完整实验流程 - 一键运行脚本
# Complete SAE Experimental Pipeline
# =============================================================================
# 
# 功能：
#   1. 训练 SAE (Sparse Autoencoder)
#   2. 分析特征并识别空间特征
#   3. 运行 Ablation 实验验证因果作用
#   4. 运行梯度归因分析
#   5. 生成所有可视化结果
#
# 使用方法：
#   bash run_complete_pipeline.sh [OPTIONS]
#
# 示例：
#   # 运行完整流程
#   bash run_complete_pipeline.sh --all
#
#   # 只运行训练和分析
#   bash run_complete_pipeline.sh --train --analyze
#
#   # 跳过训练，只做实验（使用已有checkpoint）
#   bash run_complete_pipeline.sh --checkpoint ./sae_results_new/L18_F16384_* --ablation --gradient
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# 颜色输出
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_section() {
    echo -e "${CYAN}======================================================"
    echo -e "$1"
    echo -e "======================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# =============================================================================
# 默认配置参数
# =============================================================================

# 模型和数据
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_with_prompt.json"
TEST_DATA="/home/s202507009/workspace/SA-of-LLM/SA-of-LLM/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test_with_prompt.json"

# SAE 训练参数
LAYER=18
N_FEATURES=16384
L1_COEFF=1e-5
LR=3e-4
NUM_TOKENS=100000

# 实验参数
MAX_SAMPLES=100
TOP_K=50
ABLATION_VERSION="simple"  # simple 或 full
GRADIENT_METHOD="grad_x_act"  # grad_x_act, grad_norm, integrated_gradients

# 输出目录
OUTPUT_DIR="./sae_results_new"

# 运行标志
RUN_TRAIN=false
RUN_ANALYZE=false
RUN_ABLATION=false
RUN_GRADIENT=false
RUN_ALL=false
USE_EXISTING_CHECKPOINT=""

# =============================================================================
# 解析命令行参数
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --train)
            RUN_TRAIN=true
            shift
            ;;
        --analyze)
            RUN_ANALYZE=true
            shift
            ;;
        --ablation)
            RUN_ABLATION=true
            shift
            ;;
        --gradient)
            RUN_GRADIENT=true
            shift
            ;;
        --checkpoint)
            USE_EXISTING_CHECKPOINT="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --features)
            N_FEATURES="$2"
            shift 2
            ;;
        --l1)
            L1_COEFF="$2"
            shift 2
            ;;
        --samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --ablation-version)
            ABLATION_VERSION="$2"
            shift 2
            ;;
        --gradient-method)
            GRADIENT_METHOD="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash run_complete_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all                   Run complete pipeline (train + analyze + ablation + gradient)"
            echo "  --train                 Run SAE training"
            echo "  --analyze               Run feature analysis"
            echo "  --ablation              Run ablation experiment"
            echo "  --gradient              Run gradient attribution"
            echo "  --checkpoint PATH       Use existing checkpoint (skip training)"
            echo ""
            echo "Configuration:"
            echo "  --layer N               Target layer (default: 18)"
            echo "  --features N            Number of SAE features (default: 16384)"
            echo "  --l1 FLOAT             L1 coefficient (default: 1e-5)"
            echo "  --samples N             Max evaluation samples (default: 100)"
            echo "  --ablation-version V    Ablation version: simple/full (default: simple)"
            echo "  --gradient-method M     Gradient method: grad_x_act/grad_norm/integrated_gradients"
            echo "  --output DIR            Output directory (default: ./sae_results_new)"
            echo ""
            echo "Examples:"
            echo "  # Run complete pipeline"
            echo "  bash run_complete_pipeline.sh --all"
            echo ""
            echo "  # Only training and analysis"
            echo "  bash run_complete_pipeline.sh --train --analyze"
            echo ""
            echo "  # Use existing checkpoint for experiments"
            echo "  bash run_complete_pipeline.sh --checkpoint ./sae_results_new/L18_* --ablation --gradient"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 如果指定了 --all，则运行所有步骤
if [ "$RUN_ALL" = true ]; then
    RUN_TRAIN=true
    RUN_ANALYZE=true
    RUN_ABLATION=true
    RUN_GRADIENT=true
fi

# 如果没有指定任何运行标志，显示帮助
if [ "$RUN_TRAIN" = false ] && [ "$RUN_ANALYZE" = false ] && [ "$RUN_ABLATION" = false ] && [ "$RUN_GRADIENT" = false ]; then
    print_warning "No action specified. Use --help for usage information."
    echo ""
    echo "Quick start:"
    echo "  bash run_complete_pipeline.sh --all"
    exit 1
fi

# =============================================================================
# 显示配置
# =============================================================================
print_section "SAE Complete Pipeline Configuration"
echo ""
echo "Model & Data:"
echo "  Model: $MODEL_NAME"
echo "  Train Data: $TRAIN_DATA"
echo "  Test Data: $TEST_DATA"
echo ""
echo "SAE Configuration:"
echo "  Layer: $LAYER"
echo "  Features: $N_FEATURES"
echo "  L1 Coefficient: $L1_COEFF"
echo "  Learning Rate: $LR"
echo "  Training Tokens: $NUM_TOKENS"
echo ""
echo "Experiment Configuration:"
echo "  Max Eval Samples: $MAX_SAMPLES"
echo "  Ablation Version: $ABLATION_VERSION"
echo "  Gradient Method: $GRADIENT_METHOD"
echo ""
echo "Pipeline Steps:"
echo "  Training: $([ "$RUN_TRAIN" = true ] && echo "✓ Yes" || echo "✗ No")"
echo "  Analysis: $([ "$RUN_ANALYZE" = true ] && echo "✓ Yes" || echo "✗ No")"
echo "  Ablation: $([ "$RUN_ABLATION" = true ] && echo "✓ Yes" || echo "✗ No")"
echo "  Gradient Attribution: $([ "$RUN_GRADIENT" = true ] && echo "✓ Yes" || echo "✗ No")"
echo ""
if [ -n "$USE_EXISTING_CHECKPOINT" ]; then
    echo "Using existing checkpoint: $USE_EXISTING_CHECKPOINT"
    echo ""
fi

# =============================================================================
# 检查数据文件
# =============================================================================
if [ "$RUN_TRAIN" = true ]; then
    if [ ! -f "$TRAIN_DATA" ]; then
        print_error "Training data not found: $TRAIN_DATA"
        exit 1
    fi
fi

if [ ! -f "$TEST_DATA" ]; then
    print_error "Test data not found: $TEST_DATA"
    exit 1
fi

# =============================================================================
# Step 1: 训练 SAE
# =============================================================================
if [ "$RUN_TRAIN" = true ]; then
    print_section "Step 1: Training SAE"
    echo ""
    
    python train_sae_saelens.py \
        --model_name "$MODEL_NAME" \
        --train_data_file_path "$TRAIN_DATA" \
        --test_data_file_path "$TEST_DATA" \
        --layer $LAYER \
        --n_features $N_FEATURES \
        --l1_coeff $L1_COEFF \
        --lr $LR \
        --num_tokens $NUM_TOKENS \
        --output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        print_success "SAE training completed"
    else
        print_error "SAE training failed"
        exit 1
    fi
    
    # 找到最新的 checkpoint
    CHECKPOINT_DIR=$(ls -td ${OUTPUT_DIR}/L${LAYER}_F${N_FEATURES}_* 2>/dev/null | head -n1)
    
    if [ -z "$CHECKPOINT_DIR" ]; then
        print_error "No checkpoint found in $OUTPUT_DIR"
        exit 1
    fi
    
    print_info "Checkpoint saved to: $CHECKPOINT_DIR"
    echo ""
else
    # 使用已有的 checkpoint
    if [ -n "$USE_EXISTING_CHECKPOINT" ]; then
        CHECKPOINT_DIR="$USE_EXISTING_CHECKPOINT"
    else
        # 尝试找到最新的 checkpoint
        CHECKPOINT_DIR=$(ls -td ${OUTPUT_DIR}/L${LAYER}_F${N_FEATURES}_* 2>/dev/null | head -n1)
    fi
    
    if [ -z "$CHECKPOINT_DIR" ] || [ ! -d "$CHECKPOINT_DIR" ]; then
        print_error "Checkpoint not found. Please specify --checkpoint or run with --train"
        exit 1
    fi
    
    print_info "Using checkpoint: $CHECKPOINT_DIR"
    echo ""
fi

# =============================================================================
# Step 2: 分析特征
# =============================================================================
if [ "$RUN_ANALYZE" = true ]; then
    print_section "Step 2: Analyzing Features"
    echo ""
    
    # 检查是否已经完成分析
    if [ -f "$CHECKPOINT_DIR/analysis/dimension_features.json" ]; then
        print_warning "Analysis already exists. Skipping..."
        print_info "To re-run analysis, delete: $CHECKPOINT_DIR/analysis/"
    else
        python analyze_features_saelens.py \
            --checkpoint "$CHECKPOINT_DIR" \
            --top_k $TOP_K
        
        if [ $? -eq 0 ]; then
            print_success "Feature analysis completed"
        else
            print_error "Feature analysis failed"
            exit 1
        fi
    fi
    echo ""
fi

# 检查 analysis 是否存在（后续步骤需要）
if [ "$RUN_ABLATION" = true ] || [ "$RUN_GRADIENT" = true ]; then
    if [ ! -f "$CHECKPOINT_DIR/analysis/dimension_features.json" ]; then
        print_error "Analysis not found. Please run with --analyze first"
        exit 1
    fi
fi

# =============================================================================
# Step 3: Ablation 实验
# =============================================================================
if [ "$RUN_ABLATION" = true ]; then
    print_section "Step 3: Running Ablation Experiment"
    echo ""
    
    # 选择脚本
    if [ "$ABLATION_VERSION" = "simple" ]; then
        ABLATION_SCRIPT="ablate_features_simple.py"
        print_info "Using simplified version (logits-based, faster)"
    elif [ "$ABLATION_VERSION" = "full" ]; then
        ABLATION_SCRIPT="ablate_features_saelens.py"
        print_info "Using full version (generation-based)"
    elif [ "$ABLATION_VERSION" = "generation" ]; then
        ABLATION_SCRIPT="ablate_features_generation.py"
        print_info "Using generation version (actual text generation)"
    else
        print_error "Unknown ablation version: $ABLATION_VERSION"
        exit 1
    fi
    
    python "$ABLATION_SCRIPT" \
        --model_name "$MODEL_NAME" \
        --sae_checkpoint "$CHECKPOINT_DIR" \
        --eval_data_file "$TEST_DATA" \
        --max_samples $MAX_SAMPLES
    
    if [ $? -eq 0 ]; then
        print_success "Ablation experiment completed"
        
        # 可视化 ablation 结果
        echo ""
        print_info "Generating ablation visualizations..."
        
        # 找到最新的 ablation 结果文件
        ABLATION_RESULT=$(ls -t ${CHECKPOINT_DIR}/ablation_results/ablation_results_*.json 2>/dev/null | head -1)
        
        if [ -n "$ABLATION_RESULT" ]; then
            python visualize_ablation.py \
                --results_file "$ABLATION_RESULT"
            
            if [ $? -eq 0 ]; then
                print_success "Ablation visualizations generated"
            else
                print_warning "Ablation visualization failed, but continuing..."
            fi
        else
            print_warning "No ablation result file found for visualization"
        fi
    else
        print_error "Ablation experiment failed"
        exit 1
    fi
    echo ""
fi

# =============================================================================
# Step 4: 梯度归因分析
# =============================================================================
if [ "$RUN_GRADIENT" = true ]; then
    print_section "Step 4: Running Gradient Attribution Analysis"
    echo ""
    
    print_info "Attribution method: $GRADIENT_METHOD"
    
    # 运行梯度归因
    python gradient_attribution.py \
        --model_name "$MODEL_NAME" \
        --sae_checkpoint "$CHECKPOINT_DIR" \
        --eval_data_file "$TEST_DATA" \
        --max_samples $MAX_SAMPLES \
        --attribution_method $GRADIENT_METHOD \
        --device cuda:0
    
    if [ $? -eq 0 ]; then
        print_success "Gradient attribution completed"
        
        # 可视化结果
        echo ""
        print_info "Generating gradient attribution visualizations..."
        
        # 找到最新的结果文件
        GRADIENT_RESULT=$(ls -t ${CHECKPOINT_DIR}/gradient_attribution/gradient_attribution_${GRADIENT_METHOD}_*.json 2>/dev/null | head -1)
        
        if [ -n "$GRADIENT_RESULT" ]; then
            python visualize_gradient_attribution.py \
                --results_file "$GRADIENT_RESULT" \
                --sae_checkpoint "$CHECKPOINT_DIR"
            
            if [ $? -eq 0 ]; then
                print_success "Gradient attribution visualizations generated"
            else
                print_warning "Gradient visualization failed, but continuing..."
            fi
        else
            print_warning "No gradient result file found for visualization"
        fi
    else
        print_error "Gradient attribution failed"
        exit 1
    fi
    echo ""
fi

# =============================================================================
# 最终总结
# =============================================================================
print_section "PIPELINE COMPLETE!"
echo ""
print_success "All requested steps completed successfully"
echo ""
echo "Results saved to: ${CHECKPOINT_DIR}"
echo ""
echo "Generated Files:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 训练结果
if [ "$RUN_TRAIN" = true ] || [ -f "$CHECKPOINT_DIR/sae_checkpoint.pt" ]; then
    echo ""
    echo "📦 Training:"
    echo "  • SAE checkpoint: ${CHECKPOINT_DIR}/sae_checkpoint.pt"
    echo "  • Training history: ${CHECKPOINT_DIR}/training_history.json"
    echo "  • Activations: ${CHECKPOINT_DIR}/activations.pt"
fi

# 分析结果
if [ "$RUN_ANALYZE" = true ] || [ -d "$CHECKPOINT_DIR/analysis" ]; then
    echo ""
    echo "🔍 Analysis:"
    echo "  • Spatial features: ${CHECKPOINT_DIR}/analysis/dimension_features.json"
    echo "  • Feature stats: ${CHECKPOINT_DIR}/analysis/feature_stats.npz"
    echo "  • Visualizations: ${CHECKPOINT_DIR}/analysis/*.png"
fi

# Ablation 结果
if [ "$RUN_ABLATION" = true ]; then
    echo ""
    echo "🚫 Ablation:"
    echo "  • Results: ${CHECKPOINT_DIR}/ablation_results/ablation_results_*.json"
    echo "  • Summary: ${CHECKPOINT_DIR}/ablation_results/ablation_summary_*.txt"
    echo "  • Figures: ${CHECKPOINT_DIR}/ablation_results/*.png"
fi

# 梯度归因结果
if [ "$RUN_GRADIENT" = true ]; then
    echo ""
    echo "🎯 Gradient Attribution:"
    echo "  • Results: ${CHECKPOINT_DIR}/gradient_attribution/gradient_attribution_*.json"
    echo "  • Summary: ${CHECKPOINT_DIR}/gradient_attribution/gradient_summary_*.txt"
    echo "  • Figures: ${CHECKPOINT_DIR}/gradient_attribution/*.png"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next Steps:"
echo "  1. Review spatial features:"
echo "     cat ${CHECKPOINT_DIR}/analysis/dimension_features.json"
echo ""

if [ "$RUN_ABLATION" = true ]; then
    echo "  2. Check ablation results:"
    echo "     cat ${CHECKPOINT_DIR}/ablation_results/ablation_summary_*.txt"
    echo ""
fi

if [ "$RUN_GRADIENT" = true ]; then
    echo "  3. Review gradient attribution:"
    echo "     cat ${CHECKPOINT_DIR}/gradient_attribution/gradient_summary_*.txt"
    echo ""
fi

echo "  4. View visualizations:"
echo "     open ${CHECKPOINT_DIR}/**/*.png"
echo ""

# 提供额外的实验建议
if [ "$RUN_TRAIN" = true ] && [ "$RUN_ABLATION" = false ] && [ "$RUN_GRADIENT" = false ]; then
    echo ""
    print_info "Run experiments on this checkpoint:"
    echo "  bash run_complete_pipeline.sh --checkpoint \"$CHECKPOINT_DIR\" --ablation --gradient"
    echo ""
fi

if [ "$RUN_GRADIENT" = true ] && [ "$GRADIENT_METHOD" = "grad_x_act" ]; then
    echo ""
    print_info "Try other gradient methods:"
    echo "  bash run_complete_pipeline.sh --checkpoint \"$CHECKPOINT_DIR\" --gradient --gradient-method integrated_gradients"
    echo ""
fi

print_section "Pipeline Finished"
echo ""


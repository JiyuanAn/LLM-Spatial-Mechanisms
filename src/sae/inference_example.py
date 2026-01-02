"""
SAE 推理示例
展示如何使用训练好的 SAE 对新数据进行特征提取和分析
"""

import sys
sys.path.append("./")
sys.path.append("../../")
from config import PATHS

import torch
import json
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from sae_model import SparseAutoencoder

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='SAE Inference Example')
parser.add_argument("--checkpoint", "-c", type=str, required=True,
                    help="Path to SAE checkpoint")
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--prompt", "-p", type=str, default=None,
                    help="Test prompt (or use --prompts_file)")
parser.add_argument("--prompts_file", "-pf", type=str, default=None,
                    help="JSON file with test prompts")
parser.add_argument("--top_k", type=int, default=10,
                    help="Show top-k active features")
args = parser.parse_args()

# =========================
# 加载 SAE
# =========================
print("Loading SAE checkpoint...")
# PyTorch 2.6+ 需要显式设置 weights_only=False 来加载包含 numpy 的 checkpoint
checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
sae_config = checkpoint['config']

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
LAYER_IDX = sae_config['layer']

sae = SparseAutoencoder(
    d_in=sae_config['d_in'],
    n_features=sae_config['n_features'],
    l1_coefficient=sae_config['l1_coefficient'],
    dtype=torch.float32,
).to(DEVICE)
sae.load_state_dict(checkpoint['model_state_dict'])
sae.eval()

print(f"SAE loaded: {sae_config['n_features']} features from Layer {LAYER_IDX}")

# =========================
# 加载语言模型
# =========================
print("Loading language model...")
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True
)
hf_model = hf_model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    hf_model=hf_model,
    tokenizer=tokenizer,
    dtype=DTYPE,
    device=DEVICE,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    fold_value_biases=False,
)
model.eval()

print("Model loaded!")

# =========================
# 准备测试 prompts
# =========================
if args.prompt:
    test_prompts = [args.prompt]
elif args.prompts_file:
    with open(args.prompts_file, 'r') as f:
        data = json.load(f)
        test_prompts = [item['question'] for item in data[:5]]  # 取前5个示例
else:
    # 默认示例（中文）
    test_prompts = [
        "甲在乙的左边。\n乙在丙的上面。\n甲在丙的什么位置？",
        "A在B的右边。\nB在C的下面。\nA在C的什么位置？",
        "X在Y的前面。\nY在Z的左边。\nX在Z的什么位置？",
    ]

print(f"\nTesting {len(test_prompts)} prompts...")

# =========================
# 推理函数
# =========================
def extract_sae_features(prompt: str) -> dict:
    """
    提取给定 prompt 的 SAE features
    
    返回：
        - mlp_out: 原始 MLP 输出
        - features: SAE features
        - reconstructed: 重构的 MLP 输出
        - top_k_features: Top-K 激活的 features
    """
    tokens = model.to_tokens(prompt, truncate=True)
    
    with torch.no_grad():
        # 获取 MLP 输出
        _, cache = model.run_with_cache(
            tokens,
            names_filter=f"blocks.{LAYER_IDX}.hook_mlp_out"
        )
        
        mlp_out = cache[f"blocks.{LAYER_IDX}.hook_mlp_out"][0, -1]
        mlp_out = mlp_out.float().to(DEVICE)
        
        # 通过 SAE
        features = sae.encode(mlp_out)
        reconstructed = sae.decode(features)
    
    # 计算重构质量
    mse = torch.nn.functional.mse_loss(reconstructed, mlp_out).item()
    cos_sim = torch.nn.functional.cosine_similarity(mlp_out, reconstructed, dim=0).item()
    
    # 找出 top-k 激活的 features
    feature_values = features.cpu().numpy()
    top_k_indices = np.argsort(feature_values)[::-1][:args.top_k]
    top_k_values = feature_values[top_k_indices]
    
    return {
        'mlp_out': mlp_out.cpu().numpy(),
        'features': feature_values,
        'reconstructed': reconstructed.cpu().numpy(),
        'mse': mse,
        'cos_sim': cos_sim,
        'l0': (feature_values > 0).sum(),
        'top_k_indices': top_k_indices.tolist(),
        'top_k_values': top_k_values.tolist(),
    }

# =========================
# 对每个 prompt 进行推理
# =========================
print("\n" + "="*80)
print("INFERENCE RESULTS")
print("="*80)

for i, prompt in enumerate(test_prompts):
    print(f"\n[Prompt {i+1}/{len(test_prompts)}]")
    print(f"Text: {prompt[:100]}..." if len(prompt) > 100 else f"Text: {prompt}")
    print("-" * 80)
    
    result = extract_sae_features(prompt)
    
    print(f"Reconstruction Quality:")
    print(f"  MSE: {result['mse']:.6f}")
    print(f"  Cosine Similarity: {result['cos_sim']:.4f}")
    print(f"  L0 (active features): {result['l0']}")
    print(f"\nTop {args.top_k} Active Features:")
    print(f"  {'Feature':<12} {'Activation':<12}")
    print("  " + "-" * 24)
    for feat_idx, feat_val in zip(result['top_k_indices'], result['top_k_values']):
        print(f"  {feat_idx:<12} {feat_val:<12.4f}")
    print("="*80)

# =========================
# 可选：加载 feature 分析结果并解释
# =========================
checkpoint_dir = Path(args.checkpoint).parent
analysis_dir = checkpoint_dir / 'analysis'
dimension_file = analysis_dir / 'dimension_features.json'

if dimension_file.exists():
    print("\n" + "="*80)
    print("FEATURE INTERPRETATION (if available)")
    print("="*80)
    
    with open(dimension_file, 'r') as f:
        dimension_features = json.load(f)
    
    # 为每个维度创建 feature index 到标签的映射
    feature_labels = {}
    labels = {
        'X_positive': 'Right',
        'X_negative': 'Left',
        'Y_positive': 'Above',
        'Y_negative': 'Below',
        'Z_positive': 'Front',
        'Z_negative': 'Behind',
    }
    
    for key, label in labels.items():
        for feat_info in dimension_features[key]:
            feat_idx = feat_info['feature_idx']
            if feat_idx not in feature_labels:
                feature_labels[feat_idx] = []
            feature_labels[feat_idx].append({
                'label': label,
                'correlation': feat_info['correlation']
            })
    
    # 对每个 prompt 解释激活的 features
    print("\nRe-analyzing with feature labels:\n")
    
    for i, prompt in enumerate(test_prompts):
        result = extract_sae_features(prompt)
        
        print(f"[Prompt {i+1}]: {prompt[:60]}...")
        print(f"Active spatial features:")
        
        found_spatial = False
        for feat_idx, feat_val in zip(result['top_k_indices'], result['top_k_values']):
            if feat_idx in feature_labels:
                found_spatial = True
                labels_str = ", ".join([f"{l['label']}({l['correlation']:.2f})" 
                                       for l in feature_labels[feat_idx]])
                print(f"  Feature {feat_idx} (act={feat_val:.3f}): {labels_str}")
        
        if not found_spatial:
            print("  (No labeled spatial features in top-k)")
        print()
    
    print("="*80)
else:
    print("\nNote: Run analyze_features.py first to get feature interpretations")

print("\nInference complete!")
print("\nUsage examples:")
print("  # Single prompt")
print(f"  python inference_example.py -c {args.checkpoint} -p '你的问题'")
print("  # Batch from file")
print(f"  python inference_example.py -c {args.checkpoint} -pf test_data.json")


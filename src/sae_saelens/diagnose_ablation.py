"""
诊断脚本：检查为什么 spatial ablation 没有效果
"""

import sys
sys.path.append("./")
sys.path.append("../../")
from config import PATHS

import os
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Diagnose SAE ablation')
parser.add_argument("--sae_checkpoint", "-c", type=str, required=True)
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--eval_data_file", "-e", type=str, required=True)
parser.add_argument("--max_samples", type=int, default=10)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

# =========================
# 基本配置
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = args.device if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

checkpoint_dir = Path(args.sae_checkpoint)

# =========================
# 加载 SAE
# =========================
print("="*60)
print("Loading SAE checkpoint...")
print("="*60)

checkpoint_file = checkpoint_dir / 'sae_checkpoint.pt'
checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
sae_config = checkpoint['config']
sae_state = checkpoint['model_state_dict']

LAYER_IDX = sae_config['layer']
N_FEATURES = sae_config['n_features']
D_IN = sae_config['d_in']

# 提取权重
W_enc = sae_state['W_enc'].to(DEVICE)
b_enc = sae_state['b_enc'].to(DEVICE)
W_dec = sae_state['W_dec'].to(DEVICE)
b_dec = sae_state.get('b_dec', torch.zeros(D_IN)).to(DEVICE)

print(f"SAE: Layer {LAYER_IDX}, {D_IN} -> {N_FEATURES} features")

# =========================
# 加载 spatial features
# =========================
print("\n" + "="*60)
print("Loading spatial features...")
print("="*60)

analysis_dir = checkpoint_dir / "analysis"
dimension_features_file = analysis_dir / "dimension_features.json"

with open(dimension_features_file, 'r') as f:
    dimension_features = json.load(f)

spatial_feature_ids = set()
for key in dimension_features:
    for feat in dimension_features[key]:
        spatial_feature_ids.add(feat['feature_idx'])

spatial_feature_ids = sorted(list(spatial_feature_ids))
print(f"Loaded {len(spatial_feature_ids)} spatial features")

# =========================
# 加载模型
# =========================
print("\n" + "="*60)
print("Loading model...")
print("="*60)

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

HOOK_POINT = f"blocks.{LAYER_IDX}.hook_mlp_out"

# =========================
# 加载评估数据
# =========================
print("\n" + "="*60)
print("Loading evaluation data...")
print("="*60)

with open(args.eval_data_file, 'r') as f:
    eval_data = json.load(f)

if args.max_samples:
    eval_data = eval_data[:args.max_samples]

print(f"Loaded {len(eval_data)} samples")

# =========================
# 诊断函数
# =========================
def diagnose_sample(sample_idx, sample):
    """诊断单个样本的 SAE 激活"""
    prompt = sample["prompt"]
    correct_option = sample["correct_option"]
    
    print("\n" + "="*70)
    print(f"Sample {sample_idx + 1}")
    print("="*70)
    print(f"Correct answer: {correct_option}")
    print(f"Question: {sample['question'][:100]}...")
    
    # Tokenize
    tokens = model.to_tokens(prompt, truncate=True)
    
    # 获取激活
    activations_cache = []
    
    def hook_fn(act, hook):
        activations_cache.append(act.detach().cpu())
        return act
    
    with model.hooks(fwd_hooks=[(HOOK_POINT, hook_fn)]):
        with torch.no_grad():
            _ = model(tokens)
    
    # 获取最后一个token的激活
    act = activations_cache[0][0, -1, :].to(DEVICE)  # [d_in]
    
    # 通过 SAE 编码
    with torch.no_grad():
        x_centered = act - b_dec
        h = torch.nn.functional.relu(x_centered @ W_enc + b_enc)  # [n_features]
    
    # 分析激活
    h_np = h.cpu().numpy()
    
    print(f"\nSAE Feature Activations:")
    print(f"  Total features: {N_FEATURES}")
    print(f"  Active features (>0): {(h_np > 0).sum()}")
    print(f"  Active features (>0.1): {(h_np > 0.1).sum()}")
    print(f"  Active features (>1.0): {(h_np > 1.0).sum()}")
    print(f"  Max activation: {h_np.max():.4f}")
    print(f"  Mean activation: {h_np.mean():.4f}")
    
    # 检查 spatial features 的激活
    spatial_activations = h_np[spatial_feature_ids]
    print(f"\nSpatial Features Activations:")
    print(f"  Total spatial features: {len(spatial_feature_ids)}")
    print(f"  Active spatial features (>0): {(spatial_activations > 0).sum()}")
    print(f"  Active spatial features (>0.1): {(spatial_activations > 0.1).sum()}")
    print(f"  Active spatial features (>1.0): {(spatial_activations > 1.0).sum()}")
    print(f"  Max activation: {spatial_activations.max():.4f}")
    print(f"  Mean activation: {spatial_activations.mean():.4f}")
    
    # 显示最活跃的 spatial features
    spatial_active = [(spatial_feature_ids[i], spatial_activations[i]) 
                      for i in range(len(spatial_feature_ids)) 
                      if spatial_activations[i] > 0.1]
    spatial_active.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 active spatial features:")
    for feat_id, act_val in spatial_active[:10]:
        print(f"  Feature {feat_id:4d}: {act_val:.4f}")
    
    # 计算 ablation 的影响
    # 1. 原始重构
    with torch.no_grad():
        act_reconstructed = h @ W_dec + b_dec
        recon_error = ((act - act_reconstructed) ** 2).mean().item()
    
    # 2. Spatial ablation 重构
    h_spatial_ablated = h.clone()
    h_spatial_ablated[spatial_feature_ids] = 0.0
    with torch.no_grad():
        act_spatial_ablated = h_spatial_ablated @ W_dec + b_dec
        spatial_diff = ((act_reconstructed - act_spatial_ablated) ** 2).mean().item()
    
    # 3. Random ablation 重构
    available_ids = [i for i in range(N_FEATURES) if i not in spatial_feature_ids]
    random_ids = np.random.choice(available_ids, min(len(spatial_feature_ids), len(available_ids)), replace=False)
    h_random_ablated = h.clone()
    h_random_ablated[random_ids] = 0.0
    with torch.no_grad():
        act_random_ablated = h_random_ablated @ W_dec + b_dec
        random_diff = ((act_reconstructed - act_random_ablated) ** 2).mean().item()
    
    print(f"\nReconstruction Analysis:")
    print(f"  SAE reconstruction error: {recon_error:.6f}")
    print(f"  Spatial ablation impact (MSE): {spatial_diff:.6f}")
    print(f"  Random ablation impact (MSE): {random_diff:.6f}")
    
    if spatial_diff < random_diff:
        print(f"  ⚠️ WARNING: Spatial features have LESS impact than random!")
    elif spatial_diff > random_diff * 2:
        print(f"  ✓ Good: Spatial features have 2x more impact than random")
    else:
        print(f"  ~ Moderate: Spatial features have slightly more impact")
    
    return {
        'sample_idx': sample_idx,
        'n_active_features': (h_np > 0).sum(),
        'n_active_spatial': (spatial_activations > 0).sum(),
        'max_spatial_activation': spatial_activations.max(),
        'mean_spatial_activation': spatial_activations.mean(),
        'recon_error': recon_error,
        'spatial_ablation_impact': spatial_diff,
        'random_ablation_impact': random_diff,
    }

# =========================
# 运行诊断
# =========================
print("\n" + "="*60)
print("Running Diagnostics")
print("="*60)

results = []
for i, sample in enumerate(eval_data):
    result = diagnose_sample(i, sample)
    results.append(result)

# =========================
# 总结
# =========================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

avg_active_features = np.mean([r['n_active_features'] for r in results])
avg_active_spatial = np.mean([r['n_active_spatial'] for r in results])
avg_spatial_impact = np.mean([r['spatial_ablation_impact'] for r in results])
avg_random_impact = np.mean([r['random_ablation_impact'] for r in results])

print(f"\nAverage statistics across {len(results)} samples:")
print(f"  Active features: {avg_active_features:.1f} / {N_FEATURES}")
print(f"  Active spatial features: {avg_active_spatial:.1f} / {len(spatial_feature_ids)}")
print(f"  Spatial ablation impact: {avg_spatial_impact:.6f}")
print(f"  Random ablation impact: {avg_random_impact:.6f}")
print(f"  Impact ratio (spatial/random): {avg_spatial_impact/avg_random_impact:.2f}x")

if avg_spatial_impact < avg_random_impact:
    print("\n❌ PROBLEM: Spatial features have LESS impact than random features!")
    print("   Possible causes:")
    print("   1. Spatial features are not causally important")
    print("   2. Feature selection method (correlation) is not identifying causal features")
    print("   3. These features are 'descriptive' rather than 'mechanistic'")
elif avg_spatial_impact < avg_random_impact * 1.5:
    print("\n⚠️ WARNING: Spatial features have only slightly more impact than random")
    print("   The causal effect is weak")
else:
    print("\n✓ Good: Spatial features have significantly more impact than random!")

print("\n" + "="*60)



"""
SAE Feature Ablation Experiment (Simplified Version)
简化版本：使用 logits 直接预测，更快更稳定

核心思路：
- 用 SAE hook 干预 layer 8 的激活
- 测量干预后模型对空间问题回答的准确率
- 比较 spatial ablation vs random ablation vs no ablation
"""

import sys
sys.path.append("./")
sys.path.append("../../")
from config import PATHS

import os
import json
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='SAE Feature Ablation (Simple)')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--sae_checkpoint", "-c", type=str, required=True)
parser.add_argument("--eval_data_file", "-e", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default=None)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

# =========================
# 基本配置
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = args.device if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEED = args.seed

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

checkpoint_dir = Path(args.sae_checkpoint)
if args.output_dir is None:
    output_dir = checkpoint_dir / "ablation_results"
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime('%Y%m%d_%H%M%S')

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

print(f"Model loaded: {model.cfg.n_layers} layers")

HOOK_POINT = f"blocks.{LAYER_IDX}.hook_mlp_out"

# =========================
# SAE Ablator
# =========================
class SAEAblator:
    """SAE 特征干预器"""
    def __init__(self, W_enc, b_enc, W_dec, b_dec, 
                 feature_ids_to_ablate: List[int]):
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec
        self.feature_ids = feature_ids_to_ablate
    
    def __call__(self, act, hook):
        """Hook function: ablate specified features"""
        # act shape: [batch, seq, d] or [batch, d]
        original_shape = act.shape
        is_3d = len(original_shape) == 3
        
        if is_3d:
            batch, seq, d = original_shape
            act = act.reshape(-1, d)
        
        with torch.no_grad():
            # Encode
            x_centered = act - self.b_dec
            h = torch.nn.functional.relu(x_centered @ self.W_enc + self.b_enc)
            
            # Ablate
            if len(self.feature_ids) > 0:
                h[:, self.feature_ids] = 0.0
            
            # Decode
            act_new = h @ self.W_dec + self.b_dec
        
        if is_3d:
            act_new = act_new.reshape(batch, seq, d)
        
        return act_new

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

# 定义答案的关键词
ANSWER_KEYWORDS = ['left', 'right', 'above', 'below', 'front', 'behind']

# 尝试编码每个关键词（处理可能的编码错误）
ANSWER_TOKENS = {}
for word in ANSWER_KEYWORDS:
    try:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) > 0:
            ANSWER_TOKENS[word] = tokens[0]
    except Exception as e:
        print(f"Warning: Failed to encode '{word}': {e}")

print(f"\nAnswer token IDs ({len(ANSWER_TOKENS)} words):")
for ans, tok_id in ANSWER_TOKENS.items():
    print(f"  {ans}: {tok_id}")

# =========================
# 评估函数
# =========================
def extract_primary_direction(answer_text):
    """
    从复杂答案中提取主要方向
    例如: "above (2 steps)" -> "above"
          "above and front (2 steps)" -> "above" (第一个方向)
          "left" -> "left"
    """
    answer_text = answer_text.lower().strip()
    
    # 移除括号内容
    import re
    answer_text = re.sub(r'\([^)]*\)', '', answer_text).strip()
    
    # 查找第一个匹配的方向词
    for keyword in ANSWER_KEYWORDS:
        if keyword in answer_text:
            return keyword
    
    # 如果没找到，返回原始文本的第一个词
    words = answer_text.split()
    if words:
        return words[0]
    
    return answer_text

def evaluate_with_ablation(model, data, ablator=None, desc="Eval"):
    """
    评估模型在 ablation 条件下的表现
    
    使用简化的方法：
    - 在最后一个 token 的 logits 上预测
    - 选择最高概率的空间答案
    """
    correct = 0
    total = 0
    results_detail = []
    
    for sample in tqdm(data, desc=desc):
        prompt = sample["question"]
        gt_answer_raw = sample["answer"]
        
        # 提取主要方向（处理复杂答案格式）
        gt_answer = extract_primary_direction(gt_answer_raw)
        
        # Tokenize
        tokens = model.to_tokens(prompt, truncate=True)
        
        # Forward with optional hook
        if ablator is not None:
            with model.hooks(fwd_hooks=[(HOOK_POINT, ablator)]):
                with torch.no_grad():
                    logits = model(tokens)
        else:
            with torch.no_grad():
                logits = model(tokens)
        
        # Get last token logits
        last_logits = logits[0, -1, :]
        
        # Extract logits for spatial answers
        answer_probs = {}
        for ans, tok_id in ANSWER_TOKENS.items():
            answer_probs[ans] = last_logits[tok_id].item()
        
        # Predict: argmax over spatial answers
        pred_answer = max(answer_probs, key=answer_probs.get)
        
        # Check correctness
        is_correct = (pred_answer == gt_answer)
        if is_correct:
            correct += 1
        total += 1
        
        results_detail.append({
            'prompt': prompt,
            'ground_truth': gt_answer,
            'ground_truth_raw': gt_answer_raw,
            'prediction': pred_answer,
            'correct': is_correct,
            'answer_logits': answer_probs,
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'details': results_detail,
    }

# =========================
# 运行三种条件
# =========================
print("\n" + "="*60)
print("Running Ablation Experiments")
print("="*60)
print(f"\nConditions:")
print(f"  1. No ablation (baseline)")
print(f"  2. Spatial feature ablation ({len(spatial_feature_ids)} features)")
print(f"  3. Random feature ablation ({len(spatial_feature_ids)} features)")
print()

# Condition 1: No ablation
print("Condition 1: No ablation...")
results_baseline = evaluate_with_ablation(
    model, eval_data, ablator=None, desc="No ablation"
)
acc_baseline = results_baseline['accuracy']
print(f"  Accuracy: {acc_baseline:.4f} ({results_baseline['correct']}/{results_baseline['total']})")

# Condition 2: Spatial ablation
print("\nCondition 2: Spatial feature ablation...")
ablator_spatial = SAEAblator(W_enc, b_enc, W_dec, b_dec, spatial_feature_ids)
results_spatial = evaluate_with_ablation(
    model, eval_data, ablator=ablator_spatial, desc="Spatial ablation"
)
acc_spatial = results_spatial['accuracy']
print(f"  Accuracy: {acc_spatial:.4f} ({results_spatial['correct']}/{results_spatial['total']})")

# Condition 3: Random ablation
print("\nCondition 3: Random feature ablation...")
available_ids = [i for i in range(N_FEATURES) if i not in spatial_feature_ids]
random_ids = random.sample(available_ids, min(len(spatial_feature_ids), len(available_ids)))
ablator_random = SAEAblator(W_enc, b_enc, W_dec, b_dec, random_ids)
results_random = evaluate_with_ablation(
    model, eval_data, ablator=ablator_random, desc="Random ablation"
)
acc_random = results_random['accuracy']
print(f"  Accuracy: {acc_random:.4f} ({results_random['correct']}/{results_random['total']})")

# =========================
# 结果分析
# =========================
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nAccuracy:")
print(f"  Baseline (no ablation)   : {acc_baseline:.4f}")
print(f"  Spatial ablation         : {acc_spatial:.4f} (Δ = {acc_spatial - acc_baseline:+.4f})")
print(f"  Random ablation          : {acc_random:.4f} (Δ = {acc_random - acc_baseline:+.4f})")

if acc_baseline > 0:
    spatial_drop_pct = (acc_baseline - acc_spatial) / acc_baseline * 100
    random_drop_pct = (acc_baseline - acc_random) / acc_baseline * 100
    print(f"\nRelative drop:")
    print(f"  Spatial ablation         : {spatial_drop_pct:.2f}%")
    print(f"  Random ablation          : {random_drop_pct:.2f}%")
    
    # 计算效应大小 (Cohen's h for proportions)
    effect_size = abs(acc_baseline - acc_spatial) - abs(acc_baseline - acc_random)
    print(f"\nEffect size (spatial - random): {effect_size:.4f}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if acc_spatial < acc_random and acc_spatial < acc_baseline:
    print("✓ Spatial features have causal importance!")
    print("  Ablating spatial features hurts performance more than random features.")
    if acc_baseline - acc_spatial > 0.05:
        print("  Effect is substantial (>5% accuracy drop).")
elif acc_spatial < acc_baseline - 0.01:
    print("~ Spatial features may be important, but not uniquely so.")
    print("  Both spatial and random ablation hurt performance.")
else:
    print("✗ No clear evidence for causal importance of spatial features.")
    print("  Ablation has minimal effect.")

# =========================
# 保存结果
# =========================
print("\n" + "="*60)
print("Saving results...")
print("="*60)

results = {
    'timestamp': timestamp,
    'config': {
        'model_name': MODEL_NAME,
        'layer': LAYER_IDX,
        'n_features': N_FEATURES,
        'd_in': D_IN,
        'n_spatial_features': len(spatial_feature_ids),
        'n_eval_samples': len(eval_data),
    },
    'spatial_feature_ids': spatial_feature_ids,
    'random_feature_ids': random_ids,
    'results': {
        'baseline': {
            'accuracy': float(acc_baseline),
            'correct': int(results_baseline['correct']),
            'total': int(results_baseline['total']),
        },
        'spatial_ablation': {
            'accuracy': float(acc_spatial),
            'correct': int(results_spatial['correct']),
            'total': int(results_spatial['total']),
            'drop': float(acc_baseline - acc_spatial),
            'drop_pct': float((acc_baseline - acc_spatial) / acc_baseline * 100) if acc_baseline > 0 else 0,
        },
        'random_ablation': {
            'accuracy': float(acc_random),
            'correct': int(results_random['correct']),
            'total': int(results_random['total']),
            'drop': float(acc_baseline - acc_random),
            'drop_pct': float((acc_baseline - acc_random) / acc_baseline * 100) if acc_baseline > 0 else 0,
        },
    },
    'examples': {
        'baseline': results_baseline['details'][:10],
        'spatial_ablation': results_spatial['details'][:10],
        'random_ablation': results_random['details'][:10],
    }
}

results_file = output_dir / f"ablation_results_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved to: {results_file}")

# 保存简要总结
summary_file = output_dir / f"ablation_summary_{timestamp}.txt"
with open(summary_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("SAE Feature Ablation Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Layer: {LAYER_IDX}\n")
    f.write(f"Spatial features: {len(spatial_feature_ids)}\n")
    f.write(f"Eval samples: {len(eval_data)}\n\n")
    f.write("="*60 + "\n")
    f.write("Accuracy Results:\n")
    f.write("="*60 + "\n")
    f.write(f"Baseline           : {acc_baseline:.4f}\n")
    f.write(f"Spatial ablation   : {acc_spatial:.4f} ({acc_spatial - acc_baseline:+.4f})\n")
    f.write(f"Random ablation    : {acc_random:.4f} ({acc_random - acc_baseline:+.4f})\n")
    if acc_baseline > 0:
        f.write(f"\nRelative drops:\n")
        f.write(f"  Spatial: {(acc_baseline - acc_spatial) / acc_baseline * 100:.2f}%\n")
        f.write(f"  Random:  {(acc_baseline - acc_random) / acc_baseline * 100:.2f}%\n")
    f.write("\n" + "="*60 + "\n")
    f.write("Conclusion:\n")
    f.write("="*60 + "\n")
    if acc_spatial < acc_random:
        f.write("✓ Spatial features are causally important.\n")
    else:
        f.write("✗ No strong evidence for causal importance.\n")

print(f"✓ Summary saved to: {summary_file}")

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)


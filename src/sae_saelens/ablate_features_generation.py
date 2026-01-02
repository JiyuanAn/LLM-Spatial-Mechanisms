"""
SAE Feature Ablation - Generation-based Evaluation
基于实际文本生成的评估（更准确）
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
import re
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='SAE Feature Ablation (Generation-based)')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--sae_checkpoint", "-c", type=str, required=True)
parser.add_argument("--eval_data_file", "-e", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default=None)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--max_new_tokens", type=int, default=10)
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
# SimpleSAE
# =========================
class SimpleSAE:
    def __init__(self, W_enc, b_enc, W_dec, b_dec, device):
        self.W_enc = W_enc.to(device)
        self.b_enc = b_enc.to(device)
        self.W_dec = W_dec.to(device)
        self.b_dec = b_dec.to(device)
        self.device = device
    
    def encode(self, x):
        x_centered = x - self.b_dec
        pre_relu = x_centered @ self.W_enc + self.b_enc
        return torch.nn.functional.relu(pre_relu)
    
    def decode(self, h):
        return h @ self.W_dec + self.b_dec

sae = SimpleSAE(W_enc, b_enc, W_dec, b_dec, DEVICE)

# =========================
# SAE Ablator
# =========================
class SAEAblator:
    def __init__(self, sae, feature_ids_to_ablate):
        self.sae = sae
        self.feature_ids = feature_ids_to_ablate
    
    def __call__(self, act, hook):
        original_shape = act.shape
        is_3d = len(original_shape) == 3
        
        if is_3d:
            batch, seq, d = original_shape
            act = act.reshape(-1, d)
        
        with torch.no_grad():
            h = self.sae.encode(act)
            
            if len(self.feature_ids) > 0:
                h[:, self.feature_ids] = 0.0
            
            act_new = self.sae.decode(h)
        
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

# =========================
# 答案提取
# =========================
def extract_primary_direction(answer_text):
    """从答案中提取主要方向"""
    answer_text = answer_text.lower().strip()
    answer_text = re.sub(r'\([^)]*\)', '', answer_text).strip()
    
    keywords = ['left', 'right', 'above', 'below', 'front', 'behind']
    for keyword in keywords:
        if keyword in answer_text:
            return keyword
    
    words = answer_text.split()
    if words:
        return words[0]
    
    return answer_text

# =========================
# 评估函数（基于生成）
# =========================
def evaluate_with_generation(model, tokenizer, data, ablator=None, desc="Eval"):
    """
    使用实际文本生成进行评估
    """
    correct = 0
    total = 0
    results_detail = []
    
    for sample in tqdm(data, desc=desc):
        prompt = sample["question"]
        gt_answer_raw = sample["answer"]
        gt_answer = extract_primary_direction(gt_answer_raw)
        
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        
        # Generate with hook
        # 简单的贪婪解码实现
        current_ids = input_ids.clone()
        
        for _ in range(args.max_new_tokens):
            # Forward pass
            if ablator is not None:
                with model.hooks(fwd_hooks=[(HOOK_POINT, ablator)]):
                    with torch.no_grad():
                        logits = model(current_ids)
            else:
                with torch.no_grad():
                    logits = model(current_ids)
            
            # Get next token (greedy)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        output_ids = current_ids
        
        # Decode generated text
        generated_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract prediction
        pred_answer = extract_primary_direction(generated_text)
        
        # Check correctness
        is_correct = (pred_answer == gt_answer)
        if is_correct:
            correct += 1
        total += 1
        
        results_detail.append({
            'prompt': prompt[:100],
            'ground_truth': gt_answer,
            'ground_truth_raw': gt_answer_raw,
            'generated_text': generated_text,
            'prediction': pred_answer,
            'correct': is_correct,
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'details': results_detail,
    }

# =========================
# 运行实验
# =========================
print("\n" + "="*60)
print("Running Ablation Experiments (Generation-based)")
print("="*60)
print(f"\nConditions:")
print(f"  1. No ablation (baseline)")
print(f"  2. Spatial feature ablation ({len(spatial_feature_ids)} features)")
print(f"  3. Random feature ablation ({len(spatial_feature_ids)} features)")
print()

# Condition 1: No ablation
print("Condition 1: No ablation...")
results_baseline = evaluate_with_generation(
    model, tokenizer, eval_data, ablator=None, desc="No ablation"
)
acc_baseline = results_baseline['accuracy']
print(f"  Accuracy: {acc_baseline:.4f} ({results_baseline['correct']}/{results_baseline['total']})")

# Condition 2: Spatial ablation
print("\nCondition 2: Spatial feature ablation...")
ablator_spatial = SAEAblator(sae, spatial_feature_ids)
results_spatial = evaluate_with_generation(
    model, tokenizer, eval_data, ablator=ablator_spatial, desc="Spatial ablation"
)
acc_spatial = results_spatial['accuracy']
print(f"  Accuracy: {acc_spatial:.4f} ({results_spatial['correct']}/{results_spatial['total']})")

# Condition 3: Random ablation
print("\nCondition 3: Random feature ablation...")
available_ids = [i for i in range(N_FEATURES) if i not in spatial_feature_ids]
random_ids = random.sample(available_ids, min(len(spatial_feature_ids), len(available_ids)))
ablator_random = SAEAblator(sae, random_ids)
results_random = evaluate_with_generation(
    model, tokenizer, eval_data, ablator=ablator_random, desc="Random ablation"
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
    'method': 'generation-based',
    'config': {
        'model_name': MODEL_NAME,
        'layer': LAYER_IDX,
        'n_features': N_FEATURES,
        'd_in': D_IN,
        'n_spatial_features': len(spatial_feature_ids),
        'n_eval_samples': len(eval_data),
        'max_new_tokens': args.max_new_tokens,
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

results_file = output_dir / f"ablation_generation_results_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved to: {results_file}")

# 保存简要总结
summary_file = output_dir / f"ablation_generation_summary_{timestamp}.txt"
with open(summary_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("SAE Feature Ablation Results (Generation-based)\n")
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


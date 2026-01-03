"""
SAE Feature Ablation Experiment using SAELens
基于 SAELens 训练的 SAE 特征干预实验

科学目标：
- 验证 spatial features 的因果作用
- 通过 ablation 实验证明这些特征对空间推理任务是必要的
- 对比 spatial ablation vs random ablation vs no ablation
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
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='SAE Feature Ablation Experiment')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--sae_checkpoint", "-c", type=str, required=True,
                    help="Path to SAE checkpoint directory")
parser.add_argument("--eval_data_file", "-e", type=str, required=True,
                    help="Path to evaluation data file")
parser.add_argument("--output_dir", "-o", type=str, default=None,
                    help="Output directory (default: checkpoint_dir/ablation_results)")
parser.add_argument("--max_samples", type=int, default=None,
                    help="Max evaluation samples (None=all)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

# =========================
# 基本配置
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = args.device if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # 与训练时保持一致
SEED = args.seed

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 输出目录
checkpoint_dir = Path(args.sae_checkpoint)
if args.output_dir is None:
    output_dir = checkpoint_dir / "ablation_results"
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime('%Y%m%d_%H%M%S')
print(f"Experiment timestamp: {timestamp}")
print(f"Output directory: {output_dir}")

# =========================
# 加载 SAE checkpoint
# =========================
print("\n" + "="*50)
print("Loading SAE checkpoint...")
print("="*50)

checkpoint_file = checkpoint_dir / 'sae_checkpoint.pt'
if not checkpoint_file.exists():
    print(f"Error: {checkpoint_file} not found!")
    sys.exit(1)

checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
sae_config = checkpoint['config']
sae_state = checkpoint['model_state_dict']

LAYER_IDX = sae_config['layer']
N_FEATURES = sae_config['n_features']
D_IN = sae_config['d_in']

print(f"SAE Config:")
print(f"  Layer: {LAYER_IDX}")
print(f"  d_in: {D_IN}")
print(f"  n_features: {N_FEATURES}")

# 提取 SAE weights
W_enc = sae_state['W_enc']
b_enc = sae_state['b_enc']
W_dec = sae_state['W_dec']
b_dec = sae_state.get('b_dec', torch.zeros(D_IN))

print(f"  W_enc shape: {W_enc.shape}")
print(f"  W_dec shape: {W_dec.shape}")

# =========================
# 加载 spatial feature IDs
# =========================
print("\n" + "="*50)
print("Loading spatial feature IDs...")
print("="*50)

# 从 analysis 目录加载
analysis_dir = checkpoint_dir / "analysis"
dimension_features_file = analysis_dir / "dimension_features.json"

if not dimension_features_file.exists():
    print(f"Error: {dimension_features_file} not found!")
    print("Please run analyze_features_saelens.py first")
    sys.exit(1)

with open(dimension_features_file, 'r') as f:
    dimension_features = json.load(f)

# 收集所有 spatial feature IDs
spatial_feature_ids = set()
for key in dimension_features:
    for feat in dimension_features[key]:
        spatial_feature_ids.add(feat['feature_idx'])

spatial_feature_ids = sorted(list(spatial_feature_ids))
print(f"Loaded {len(spatial_feature_ids)} spatial features")

# 打印各维度特征数量
labels = {
    'X_positive': 'Right',
    'X_negative': 'Left',
    'Y_positive': 'Above',
    'Y_negative': 'Below',
    'Z_positive': 'Front',
    'Z_negative': 'Behind',
}
print("\nSpatial features by dimension:")
for key, label in labels.items():
    count = len(dimension_features[key])
    print(f"  {label:<10}: {count} features")

# =========================
# 加载模型
# =========================
print("\n" + "="*50)
print("Loading model...")
print(f"Model: {MODEL_NAME}")
print(f"Path: {MODEL_PATH}")
print("="*50)

# 先用 transformers 加载 HF 模型
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True
)
hf_model = hf_model.to(DEVICE)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("Converting to HookedTransformer...")

# 转换为 HookedTransformer
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

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model

print(f"Model loaded: {n_layers} layers, d_model={d_model}")

# 确定 hook point
# 根据训练时使用的 hook point (从 config 中读取)
HOOK_POINT = f"blocks.{LAYER_IDX}.hook_mlp_out"
print(f"Hook point: {HOOK_POINT}")

# =========================
# 定义 SAE (简化版)
# =========================
class SimpleSAE:
    """简化的 SAE，用于干预实验"""
    def __init__(self, W_enc, b_enc, W_dec, b_dec, device):
        self.W_enc = W_enc.to(device)
        self.b_enc = b_enc.to(device)
        self.W_dec = W_dec.to(device)
        self.b_dec = b_dec.to(device)
        self.device = device
    
    def encode(self, x):
        """编码：x -> h"""
        x_centered = x - self.b_dec
        pre_relu = x_centered @ self.W_enc + self.b_enc
        return torch.nn.functional.relu(pre_relu)
    
    def decode(self, h):
        """解码：h -> x_hat"""
        return h @ self.W_dec + self.b_dec
    
    def forward(self, x):
        """完整前向传播"""
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

# 创建 SAE
sae = SimpleSAE(W_enc, b_enc, W_dec, b_dec, DEVICE)
print(f"\nSAE loaded and ready for intervention")

# =========================
# 定义 SAE Ablator (核心部分)
# =========================
class SAEAblator:
    """
    SAE 特征干预器
    
    三种模式：
    - 'spatial': ablate spatial features
    - 'random': ablate random features (same count)
    - 'none': no ablation (baseline)
    """
    def __init__(self, sae: SimpleSAE, spatial_feature_ids: List[int], 
                 mode: str = "spatial", n_features: int = None):
        """
        Args:
            sae: SimpleSAE instance
            spatial_feature_ids: list of spatial feature indices
            mode: 'spatial', 'random', or 'none'
            n_features: total number of features (for random selection)
        """
        self.sae = sae
        self.mode = mode
        self.spatial_ids = spatial_feature_ids
        self.n_features = n_features
        self.k = len(spatial_feature_ids)
        
        # 预先生成随机 feature IDs (固定随机种子)
        if mode == "random":
            available_ids = [i for i in range(n_features) if i not in spatial_feature_ids]
            self.random_ids = random.sample(available_ids, min(self.k, len(available_ids)))
            print(f"  Random ablation: {len(self.random_ids)} features selected")
    
    def __call__(self, act, hook):
        """
        Hook function
        Args:
            act: activation tensor [batch, seq, d_model] or [batch, d_model]
            hook: hook object (not used here)
        
        Returns:
            modified activation
        """
        original_shape = act.shape
        
        # 处理不同形状：[batch, seq, d] or [batch, d]
        if len(original_shape) == 3:
            batch_size, seq_len, d = original_shape
            act_flat = act.reshape(-1, d)  # [batch*seq, d]
        else:
            act_flat = act
        
        with torch.no_grad():
            # Step 1: Encode
            h = self.sae.encode(act_flat)
            
            # Step 2: Ablate features
            if self.mode == "spatial":
                h[:, self.spatial_ids] = 0.0
            elif self.mode == "random":
                h[:, self.random_ids] = 0.0
            elif self.mode == "none":
                pass  # no ablation
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            # Step 3: Decode
            act_hat = self.sae.decode(h)
        
        # 恢复原始形状
        if len(original_shape) == 3:
            act_hat = act_hat.reshape(batch_size, seq_len, d)
        
        return act_hat

# =========================
# 加载评估数据
# =========================
print("\n" + "="*50)
print("Loading evaluation data...")
print("="*50)

with open(args.eval_data_file, 'r') as f:
    eval_data_raw = json.load(f)

if args.max_samples is not None:
    eval_data_raw = eval_data_raw[:args.max_samples]

print(f"Loaded {len(eval_data_raw)} evaluation samples")

# =========================
# 评估函数
# =========================
def extract_spatial_answer(text: str) -> Optional[str]:
    """
    从模型输出中提取空间答案
    
    预期格式: "left", "right", "above", "below", "front", "behind"
    """
    text = text.lower().strip()
    
    # 定义关键词
    spatial_keywords = {
        'left': ['left'],
        'right': ['right'],
        'above': ['above', 'over'],
        'below': ['below', 'under'],
        'front': ['front', 'forward'],
        'behind': ['behind', 'back'],
    }
    
    # 简单匹配：找第一个出现的关键词
    for answer, keywords in spatial_keywords.items():
        for kw in keywords:
            if kw in text:
                return answer
    
    # 如果没有匹配到，返回原始文本（第一个词）
    words = text.split()
    if words:
        return words[0]
    
    return None

def run_eval(model: HookedTransformer, data: List[Dict], 
             mode: str, ablator: Optional[SAEAblator] = None,
             verbose: bool = False) -> Dict:
    """
    运行评估
    
    Args:
        model: HookedTransformer
        data: evaluation data
        mode: 'spatial', 'random', or 'none'
        ablator: SAEAblator instance (if None, create new one)
        verbose: print detailed results
    
    Returns:
        results dict with accuracy and examples
    """
    if ablator is None:
        ablator = SAEAblator(sae, spatial_feature_ids, mode=mode, n_features=N_FEATURES)
    
    correct = 0
    total = 0
    examples = []
    
    for i, sample in enumerate(tqdm(data, desc=f"Eval ({mode})")):
        prompt = sample["prompt"]
        gt = sample["answer"].lower().strip()  # ground truth
        
        # Tokenize
        tokens = model.to_tokens(prompt, truncate=True)
        
        # Run with hook
        with model.hooks(fwd_hooks=[(HOOK_POINT, ablator)]):
            with torch.no_grad():
                logits = model(tokens)
        
        # Generate continuation (greedy decoding)
        # 简单方法：取 logits 的最后一个 token，生成几个 tokens
        max_new_tokens = 5
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            generated_tokens.append(next_token)
            
            # 如果遇到 eos token，停止
            if next_token == tokenizer.eos_token_id:
                break
            
            # 继续生成
            next_token_tensor = torch.tensor([[next_token]], device=DEVICE)
            with model.hooks(fwd_hooks=[(HOOK_POINT, ablator)]):
                with torch.no_grad():
                    logits = model(next_token_tensor)
        
        # Decode
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        pred = extract_spatial_answer(generated_text)
        
        # Check correctness
        is_correct = (pred == gt)
        if is_correct:
            correct += 1
        total += 1
        
        # Save example
        if i < 10 or (verbose and not is_correct):  # 保存前10个或错误样本
            examples.append({
                'prompt': prompt,
                'ground_truth': gt,
                'prediction': pred,
                'generated_text': generated_text,
                'correct': is_correct,
            })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'examples': examples,
    }

# =========================
# 主实验：三种条件
# =========================
print("\n" + "="*50)
print("Running SAE Feature Ablation Experiment")
print("="*50)
print(f"\nConditions:")
print(f"  1. No ablation (baseline)")
print(f"  2. Spatial feature ablation ({len(spatial_feature_ids)} features)")
print(f"  3. Random feature ablation ({len(spatial_feature_ids)} features)")
print()

# Condition 1: No ablation
print("Running Condition 1: No ablation...")
results_none = run_eval(model, eval_data_raw, mode="none", verbose=False)
acc_none = results_none['accuracy']
print(f"✓ No ablation accuracy: {acc_none:.4f} ({results_none['correct']}/{results_none['total']})")

# Condition 2: Spatial ablation
print("\nRunning Condition 2: Spatial feature ablation...")
results_spatial = run_eval(model, eval_data_raw, mode="spatial", verbose=False)
acc_spatial = results_spatial['accuracy']
print(f"✓ Spatial ablation accuracy: {acc_spatial:.4f} ({results_spatial['correct']}/{results_spatial['total']})")

# Condition 3: Random ablation
print("\nRunning Condition 3: Random feature ablation...")
results_random = run_eval(model, eval_data_raw, mode="random", verbose=False)
acc_random = results_random['accuracy']
print(f"✓ Random ablation accuracy: {acc_random:.4f} ({results_random['correct']}/{results_random['total']})")

# =========================
# 结果总结
# =========================
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"\nNo ablation        : {acc_none:.4f}")
print(f"Spatial ablation   : {acc_spatial:.4f} (Δ = {acc_spatial - acc_none:+.4f})")
print(f"Random ablation    : {acc_random:.4f} (Δ = {acc_random - acc_none:+.4f})")

# 计算相对下降
if acc_none > 0:
    spatial_drop = (acc_none - acc_spatial) / acc_none * 100
    random_drop = (acc_none - acc_random) / acc_none * 100
    print(f"\nRelative accuracy drop:")
    print(f"  Spatial ablation: {spatial_drop:.2f}%")
    print(f"  Random ablation: {random_drop:.2f}%")

# 科学结论
print("\n" + "="*50)
print("SCIENTIFIC INTERPRETATION")
print("="*50)

if acc_spatial < acc_random:
    print("✓ Spatial features are causally important!")
    print("  Ablating spatial features hurts more than random features.")
else:
    print("✗ No clear causal evidence")
    print("  Spatial ablation does not selectively hurt performance.")

# =========================
# 保存结果
# =========================
print("\n" + "="*50)
print("Saving results...")
print("="*50)

results = {
    'timestamp': timestamp,
    'config': {
        'model_name': MODEL_NAME,
        'layer': LAYER_IDX,
        'n_features': N_FEATURES,
        'd_in': D_IN,
        'hook_point': HOOK_POINT,
        'n_spatial_features': len(spatial_feature_ids),
        'eval_samples': len(eval_data_raw),
    },
    'spatial_features': {
        'ids': spatial_feature_ids,
        'by_dimension': {k: len(v) for k, v in dimension_features.items()},
    },
    'results': {
        'no_ablation': {
            'accuracy': acc_none,
            'correct': results_none['correct'],
            'total': results_none['total'],
        },
        'spatial_ablation': {
            'accuracy': acc_spatial,
            'correct': results_spatial['correct'],
            'total': results_spatial['total'],
            'accuracy_drop': float(acc_none - acc_spatial),
            'relative_drop_pct': float((acc_none - acc_spatial) / acc_none * 100) if acc_none > 0 else 0,
        },
        'random_ablation': {
            'accuracy': acc_random,
            'correct': results_random['correct'],
            'total': results_random['total'],
            'accuracy_drop': float(acc_none - acc_random),
            'relative_drop_pct': float((acc_none - acc_random) / acc_none * 100) if acc_none > 0 else 0,
        },
    },
    'examples': {
        'no_ablation': results_none['examples'],
        'spatial_ablation': results_spatial['examples'],
        'random_ablation': results_random['examples'],
    }
}

# 保存 JSON
results_file = output_dir / f"ablation_results_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved to: {results_file}")

# 保存简要总结
summary_file = output_dir / f"ablation_summary_{timestamp}.txt"
with open(summary_file, 'w') as f:
    f.write("="*50 + "\n")
    f.write("SAE Feature Ablation Results\n")
    f.write("="*50 + "\n")
    f.write(f"\nModel: {MODEL_NAME}\n")
    f.write(f"Layer: {LAYER_IDX}\n")
    f.write(f"Spatial features: {len(spatial_feature_ids)}\n")
    f.write(f"Eval samples: {len(eval_data_raw)}\n")
    f.write("\n" + "="*50 + "\n")
    f.write("Results:\n")
    f.write("="*50 + "\n")
    f.write(f"No ablation        : {acc_none:.4f}\n")
    f.write(f"Spatial ablation   : {acc_spatial:.4f} (Δ = {acc_spatial - acc_none:+.4f})\n")
    f.write(f"Random ablation    : {acc_random:.4f} (Δ = {acc_random - acc_none:+.4f})\n")
    if acc_none > 0:
        f.write(f"\nRelative drop:\n")
        f.write(f"  Spatial: {(acc_none - acc_spatial) / acc_none * 100:.2f}%\n")
        f.write(f"  Random: {(acc_none - acc_random) / acc_none * 100:.2f}%\n")

print(f"✓ Summary saved to: {summary_file}")

print("\n" + "="*50)
print("EXPERIMENT COMPLETE")
print("="*50)


"""
加载训练好的 SAE 并分析特征
"""
import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm

from transformer_lens import HookedTransformer
from train_sae import (
    SparseAutoencoder,
    load_jsonl,
    get_layer_mlp_out_activations,
    batch_iter,
    load_hooked_transformer,
)


def load_sae_checkpoint(checkpoint_path: str) -> Tuple[SparseAutoencoder, Dict[str, Any]]:
    """加载 SAE 检查点"""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["cfg"]
    
    # 从配置中获取模型维度
    # 注意：需要先加载实际模型来获取 d_model
    # 这里我们从 state_dict 推断
    state_dict = ckpt["state_dict"]
    d_in = state_dict["W_enc"].shape[1]
    n_features = state_dict["W_enc"].shape[0]
    
    sae = SparseAutoencoder(d_in=d_in, n_features=n_features)
    sae.load_state_dict(state_dict)
    sae.eval()
    
    return sae, cfg


@torch.no_grad()
def analyze_feature_activations(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    data: List[Dict[str, Any]],
    layer: int,
    token_strategy: str,
    max_prompt_tokens: int,
    batch_size: int = 8,
    device: str = "cuda"
) -> Dict[int, Dict[str, Any]]:
    """
    分析所有特征的激活统计
    返回每个特征的统计信息：
    - mean_activation: 平均激活值
    - activation_frequency: 激活频率（非零比例）
    - max_activation: 最大激活值
    """
    sae = sae.to(device)
    n_features = sae.n_features
    
    # 初始化统计
    total_samples = 0
    activation_sums = torch.zeros(n_features, device=device)
    activation_counts = torch.zeros(n_features, device=device)
    max_activations = torch.zeros(n_features, device=device)
    
    for batch in tqdm(batch_iter(data, batch_size, shuffle=False), desc="Analyzing features"):
        prompts = [x["prompt"] for x in batch]
        
        # 提取激活
        x = get_layer_mlp_out_activations(
            model, prompts, layer, token_strategy, max_prompt_tokens
        ).float().to(device)
        
        # 编码
        h = sae.encode(x)  # [B, n_features]
        
        # 更新统计
        activation_sums += h.sum(dim=0)
        activation_counts += (h > 0).sum(dim=0).float()
        max_activations = torch.maximum(max_activations, h.max(dim=0).values)
        total_samples += h.shape[0]
    
    # 计算统计
    stats = {}
    for i in range(n_features):
        stats[i] = {
            "mean_activation": float(activation_sums[i].item() / total_samples),
            "activation_frequency": float(activation_counts[i].item() / total_samples),
            "max_activation": float(max_activations[i].item()),
        }
    
    return stats


@torch.no_grad()
def get_top_examples_for_feature(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    data: List[Dict[str, Any]],
    feature_id: int,
    layer: int,
    token_strategy: str,
    max_prompt_tokens: int,
    top_k: int = 10,
    batch_size: int = 8,
    device: str = "cuda"
) -> List[Tuple[float, str]]:
    """获取特定特征激活最高的样本"""
    sae = sae.to(device)
    scored = []
    
    for batch in batch_iter(data, batch_size, shuffle=False):
        prompts = [x["prompt"] for x in batch]
        x = get_layer_mlp_out_activations(
            model, prompts, layer, token_strategy, max_prompt_tokens
        ).float().to(device)
        h = sae.encode(x)  # [B, n_features]
        vals = h[:, feature_id].cpu().numpy().tolist()
        for v, p in zip(vals, prompts):
            scored.append((float(v), p))
    
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_k]


def main():
    parser = argparse.ArgumentParser(description="分析训练好的 SAE 特征")
    parser.add_argument("--checkpoint", type=str, required=True, help="SAE 检查点路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="数据文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径")
    
    parser.add_argument("--analyze_all", action="store_true", help="分析所有特征的统计信息")
    parser.add_argument("--feature_id", type=int, default=None, help="分析特定特征的 top 样本")
    parser.add_argument("--top_k", type=int, default=10, help="显示 top-k 样本")
    
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    
    args = parser.parse_args()
    
    # 加载 SAE
    print(f"Loading SAE from: {args.checkpoint}")
    sae, cfg = load_sae_checkpoint(args.checkpoint)
    sae = sae.to(args.device)
    print(f"SAE: {sae.d_in} -> {sae.n_features} features")
    
    # 加载模型
    print(f"Loading model from: {args.model_path}")
    model = load_hooked_transformer(
        args.model_path,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    
    # 加载数据
    print(f"Loading data from: {args.data_path}")
    data = load_jsonl(args.data_path)
    print(f"Loaded {len(data)} samples")
    
    layer = cfg.get("layer", 8)
    token_strategy = cfg.get("token_strategy", "last")
    max_prompt_tokens = cfg.get("max_prompt_tokens", 2048)
    
    if args.analyze_all:
        print("\n" + "="*60)
        print("Analyzing all features...")
        print("="*60)
        
        stats = analyze_feature_activations(
            model, sae, data, layer, token_strategy, max_prompt_tokens,
            batch_size=args.batch_size, device=args.device
        )
        
        # 按激活频率排序
        sorted_features = sorted(
            stats.items(),
            key=lambda x: x[1]["activation_frequency"],
            reverse=True
        )
        
        print(f"\nTop 20 most frequently activated features:")
        print(f"{'Feature':<10} {'Freq':<12} {'Mean':<12} {'Max':<12}")
        print("-" * 50)
        for feat_id, stat in sorted_features[:20]:
            print(f"{feat_id:<10} {stat['activation_frequency']:<12.4f} "
                  f"{stat['mean_activation']:<12.6f} {stat['max_activation']:<12.4f}")
        
        print(f"\nTop 20 features with highest mean activation:")
        sorted_by_mean = sorted(
            stats.items(),
            key=lambda x: x[1]["mean_activation"],
            reverse=True
        )
        print(f"{'Feature':<10} {'Mean':<12} {'Freq':<12} {'Max':<12}")
        print("-" * 50)
        for feat_id, stat in sorted_by_mean[:20]:
            print(f"{feat_id:<10} {stat['mean_activation']:<12.6f} "
                  f"{stat['activation_frequency']:<12.4f} {stat['max_activation']:<12.4f}")
        
        # 保存统计
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"\nStatistics saved to: {args.output}")
    
    if args.feature_id is not None:
        print("\n" + "="*60)
        print(f"Analyzing Feature {args.feature_id}")
        print("="*60)
        
        top_examples = get_top_examples_for_feature(
            model, sae, data, args.feature_id, layer, token_strategy,
            max_prompt_tokens, top_k=args.top_k, batch_size=args.batch_size,
            device=args.device
        )
        
        print(f"\nTop {args.top_k} examples with highest activation:")
        print(f"{'Score':<10} {'Prompt'}")
        print("-" * 80)
        for score, prompt in top_examples:
            # 截断并清理 prompt
            clean_prompt = prompt.replace("\n", " ").strip()
            if len(clean_prompt) > 100:
                clean_prompt = clean_prompt[:97] + "..."
            print(f"{score:<10.4f} {clean_prompt}")
        
        # 如果指定输出文件，保存详细结果
        if args.output:
            output_data = {
                "feature_id": args.feature_id,
                "top_examples": [
                    {"score": score, "prompt": prompt}
                    for score, prompt in top_examples
                ]
            }
            output_path = args.output if args.output.endswith(".json") else f"{args.output}_feature_{args.feature_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
    
    if not args.analyze_all and args.feature_id is None:
        print("\nPlease specify either --analyze_all or --feature_id <ID>")
        print("Examples:")
        print("  python analyze_features.py --checkpoint sae.pt --model_path ... --data_path ... --analyze_all")
        print("  python analyze_features.py --checkpoint sae.pt --model_path ... --data_path ... --feature_id 42")


if __name__ == "__main__":
    main()


import os
import json
import math
import random
import argparse
from dataclasses import asdict, dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer


# --------------------------
# Config
# --------------------------
@dataclass
class TrainConfig:
    model_path: str
    data_path: str
    out_dir: str

    layer: int = 8
    hook_name: str = "hook_mlp_out"  # fixed by request
    token_strategy: str = "last"     # last | question_mark | answer_prefix

    # SAE hyperparams
    n_features: int = 4096
    l1_coef: float = 3e-4
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # training
    batch_size: int = 4
    max_steps: int = 5000
    log_every: int = 50
    eval_every: int = 500
    seed: int = 42

    # misc
    device: str = "cuda"
    dtype: str = "float16"  # model dtype
    max_prompt_tokens: int = 2048
    num_workers: int = 0


# --------------------------
# Dataset (JSONL or JSON)
# JSONL: Each line: {"prompt": "...", ...}
# JSON: Array of objects with "prompt" or "question" field
# --------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL or JSON file.
    Auto-detects format and normalizes field names.
    """
    data = []
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    # Try to parse as complete JSON first (array format)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            data = parsed
        elif isinstance(parsed, dict):
            # Single object, wrap in list
            data = [parsed]
        else:
            raise ValueError(f"Unexpected JSON type: {type(parsed)}")
    except json.JSONDecodeError:
        # Fall back to JSONL format (line by line)
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {line[:100]}... Error: {e}")
                continue
    
    # Normalize field names: ensure "prompt" field exists
    for item in data:
        if "prompt" not in item:
            # Try common alternatives
            if "question" in item:
                item["prompt"] = item["question"]
            elif "text" in item:
                item["prompt"] = item["text"]
            elif "input" in item:
                item["prompt"] = item["input"]
            else:
                raise ValueError(f"Data item missing 'prompt' field and no alternative found: {list(item.keys())}")
    
    return data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_hooked_transformer(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> HookedTransformer:
    """
    Load HookedTransformer from either official model name or local path.
    
    For local paths, this function:
    1. Loads the model using transformers
    2. Converts it to HookedTransformer with the appropriate official name
    
    Args:
        model_path: Official model name (e.g., "Qwen/Qwen2.5-7B-Instruct") 
                    or local path (e.g., "/path/to/model")
        device: Device to load model on
        dtype: Data type for model weights
    
    Returns:
        HookedTransformer instance
    """
    import os
    
    # Check if it's a local path
    is_local = os.path.exists(model_path) and os.path.isdir(model_path)
    
    if is_local:
        print(f"Loading model from local path: {model_path}")
        
        # Load with transformers first
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        hf_model = hf_model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Infer official model name from local path or config
        # Try to read from config.json
        config_path = os.path.join(model_path, "config.json")
        official_name = None
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                # Common fields that might contain the model name
                for key in ["_name_or_path", "model_type", "architectures"]:
                    if key in config:
                        value = config[key]
                        if isinstance(value, list):
                            value = value[0] if value else None
                        if value and ("qwen" in value.lower() or "Qwen" in value):
                            # Try to construct official name
                            if "2.5" in model_path and "7B" in model_path:
                                if "Instruct" in model_path or "instruct" in model_path:
                                    official_name = "Qwen/Qwen2.5-7B-Instruct"
                                else:
                                    official_name = "Qwen/Qwen2.5-7B"
                            break
        
        # Fallback: try to infer from path
        if official_name is None:
            path_lower = model_path.lower()
            if "qwen2.5" in path_lower or "qwen-2.5" in path_lower:
                if "7b" in path_lower:
                    official_name = "Qwen/Qwen2.5-7B-Instruct" if "instruct" in path_lower else "Qwen/Qwen2.5-7B"
                elif "14b" in path_lower:
                    official_name = "Qwen/Qwen2.5-14B-Instruct" if "instruct" in path_lower else "Qwen/Qwen2.5-14B"
                elif "3b" in path_lower:
                    official_name = "Qwen/Qwen2.5-3B-Instruct" if "instruct" in path_lower else "Qwen/Qwen2.5-3B"
            elif "qwen2" in path_lower:
                if "7b" in path_lower:
                    official_name = "Qwen/Qwen2-7B-Instruct" if "instruct" in path_lower else "Qwen/Qwen2-7B"
        
        if official_name is None:
            raise ValueError(
                f"Could not infer official model name from local path: {model_path}\n"
                f"Please use an official model name instead, or ensure the path contains model info."
            )
        
        print(f"Converting to HookedTransformer (using official name: {official_name})...")
        model = HookedTransformer.from_pretrained(
            official_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )
    else:
        # Load directly from official name
        print(f"Loading model from HuggingFace: {model_path}")
        model = HookedTransformer.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
        )
    
    model.eval()
    return model


# --------------------------
# SAE Model
# --------------------------
class SparseAutoencoder(nn.Module):
    """
    SAE: x -> h = ReLU(W_enc x + b_enc)
         x_hat = W_dec h + b_dec

    Loss = MSE(x_hat, x) + l1_coef * mean(|h|)
    """
    def __init__(self, d_in: int, n_features: int):
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features

        self.W_enc = nn.Parameter(torch.empty(n_features, d_in))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        self.W_dec = nn.Parameter(torch.empty(d_in, n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming init for encoder; decoder small init
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.normal_(self.W_dec, std=0.02)
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_in]
        h = F.relu(F.linear(x, self.W_enc, self.b_enc))  # [B, n_features]
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        x_hat = F.linear(h, self.W_dec, self.b_dec)      # [B, d_in]
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h


# --------------------------
# Token index strategies
# --------------------------
def _find_token_subseq(tokens_1d: torch.Tensor, subseq_1d: torch.Tensor) -> Optional[int]:
    """
    Find first occurrence of subseq in tokens_1d.
    Return start index or None.
    """
    if subseq_1d.numel() == 0 or tokens_1d.numel() < subseq_1d.numel():
        return None
    # naive scan (fine for short subseq)
    for i in range(tokens_1d.numel() - subseq_1d.numel() + 1):
        if torch.equal(tokens_1d[i:i + subseq_1d.numel()], subseq_1d):
            return int(i)
    return None


def pick_token_index(
    model: HookedTransformer,
    tokens: torch.Tensor,                 # [B, T]
    prompts: List[str],
    strategy: str = "last"
) -> torch.Tensor:
    """
    Return idx: [B] token positions to read activations from.
    strategy:
      - "last": last non-pad token
      - "question_mark": token right before the first "?" (or the "?" token if not found)
      - "answer_prefix": token right before the substring "Answer:" (or last if not found)
    """
    B, T = tokens.shape
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        # Qwen tokenizer may not set pad; TransformerLens usually pads with 0.
        pad_id = 0

    idx = torch.zeros(B, dtype=torch.long, device=tokens.device)

    if strategy == "last":
        # last non-pad token
        for b in range(B):
            row = tokens[b]
            nonpad = (row != pad_id).nonzero(as_tuple=False)
            idx[b] = nonpad[-1].item() if nonpad.numel() > 0 else T - 1
        return idx

    # Pre-tokenize markers once (as token subsequences)
    if strategy == "question_mark":
        marker_text = "?"
    elif strategy == "answer_prefix":
        marker_text = "Answer:"
    else:
        raise ValueError(f"Unknown token_strategy: {strategy}")

    marker_ids = model.to_tokens(marker_text, prepend_bos=False).squeeze(0)  # [m]
    marker_ids = marker_ids.to(tokens.device)

    for b in range(B):
        row = tokens[b]
        # remove pads for searching
        nonpad = (row != pad_id).nonzero(as_tuple=False)
        if nonpad.numel() == 0:
            idx[b] = T - 1
            continue
        end = nonpad[-1].item() + 1
        row_np = row[:end]

        start = _find_token_subseq(row_np, marker_ids)
        if start is None:
            idx[b] = end - 1
        else:
            if strategy == "question_mark":
                # choose token index = position of "?" token (or the token just before it, as you used in probing)
                q_pos = start  # where "?" begins
                idx[b] = max(q_pos, 0)
            else:
                # choose token right before "Answer:" begins
                idx[b] = max(start - 1, 0)

    return idx


# --------------------------
# Activation extraction
# --------------------------
@torch.no_grad()
def get_layer_mlp_out_activations(
    model: HookedTransformer,
    prompts: List[str],
    layer: int,
    token_strategy: str,
    max_prompt_tokens: int
) -> torch.Tensor:
    """
    Returns activations: [B, d_mlp_out] where d_mlp_out == d_model
    For hook_mlp_out in TransformerLens, tensor is [B, T, d_model]
    We'll select one token per prompt according to token_strategy.
    """
    tokens = model.to_tokens(prompts, truncate=True)  # [B, T]
    if tokens.shape[1] > max_prompt_tokens:
        tokens = tokens[:, :max_prompt_tokens]

    hook = f"blocks.{layer}.hook_mlp_out"
    _, cache = model.run_with_cache(tokens, names_filter=hook)

    acts = cache[hook]  # [B, T, d_model]
    idx = pick_token_index(model, tokens, prompts, strategy=token_strategy)  # [B]
    # gather: acts[b, idx[b], :]
    B = acts.shape[0]
    gathered = acts[torch.arange(B, device=acts.device), idx, :]  # [B, d_model]
    return gathered


# --------------------------
# Train / Eval
# --------------------------
def batch_iter(data: List[Dict[str, Any]], batch_size: int, shuffle: bool = True):
    idxs = list(range(len(data)))
    if shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        batch = [data[j] for j in idxs[i:i + batch_size]]
        yield batch


def save_checkpoint(out_dir: str, sae: SparseAutoencoder, cfg: TrainConfig, step: int):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "step": step,
        "cfg": asdict(cfg),
        "state_dict": sae.state_dict(),
    }
    path = os.path.join(out_dir, f"sae_step_{step}.pt")
    torch.save(ckpt, path)


@torch.no_grad()
def eval_recon(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    data: List[Dict[str, Any]],
    cfg: TrainConfig,
    n_batches: int = 20
) -> Dict[str, float]:
    sae.eval()
    losses = []
    sparsities = []
    it = batch_iter(data, cfg.batch_size, shuffle=True)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        prompts = [x["prompt"] for x in batch]
        x = get_layer_mlp_out_activations(
            model, prompts, cfg.layer, cfg.token_strategy, cfg.max_prompt_tokens
        ).float()  # [B, d_model] fp32
        x_hat, h = sae(x)
        mse = F.mse_loss(x_hat, x).item()
        l1 = h.abs().mean().item()
        losses.append(mse)
        sparsities.append(l1)
    return {
        "mse": float(np.mean(losses)) if losses else float("nan"),
        "mean_abs_h": float(np.mean(sparsities)) if sparsities else float("nan"),
    }


@torch.no_grad()
def feature_top_examples(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    data: List[Dict[str, Any]],
    cfg: TrainConfig,
    feature_id: int,
    top_k: int = 10,
    sample_n: int = 2000
) -> List[Tuple[float, str]]:
    """
    Return top-k prompts with highest activation on a given feature.
    """
    sae.eval()
    sample = data if len(data) <= sample_n else random.sample(data, sample_n)
    scored = []
    for batch in batch_iter(sample, cfg.batch_size, shuffle=False):
        prompts = [x["prompt"] for x in batch]
        x = get_layer_mlp_out_activations(
            model, prompts, cfg.layer, cfg.token_strategy, cfg.max_prompt_tokens
        ).float()
        h = sae.encode(x)  # [B, n_features]
        vals = h[:, feature_id].detach().cpu().numpy().tolist()
        for v, p in zip(vals, prompts):
            scored.append((float(v), p))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="JSONL file with {\"prompt\": ...}")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--token_strategy", type=str, default="last",
                        choices=["last", "question_mark", "answer_prefix"])

    parser.add_argument("--n_features", type=int, default=4096)
    parser.add_argument("--l1_coef", type=float, default=3e-4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--max_prompt_tokens", type=int, default=2048)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        model_path=args.model_path,
        data_path=args.data_path,
        out_dir=args.out_dir,
        layer=args.layer,
        token_strategy=args.token_strategy,
        n_features=args.n_features,
        l1_coef=args.l1_coef,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        max_prompt_tokens=args.max_prompt_tokens,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load data
    data = load_jsonl(cfg.data_path)
    if len(data) < 10:
        raise ValueError("Dataset too small. Provide at least ~hundreds prompts for SAE training.")
    # Split train/val
    random.shuffle(data)
    split = int(0.95 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # Load model
    print(f"Loading HookedTransformer from: {cfg.model_path}")
    model = load_hooked_transformer(
        cfg.model_path,
        device=cfg.device,
        dtype=getattr(torch, cfg.dtype),
    )

    d_model = model.cfg.d_model
    print(f"Model d_model={d_model}, n_layers={model.cfg.n_layers}")
    print(f"Training SAE on blocks.{cfg.layer}.hook_mlp_out, token_strategy={cfg.token_strategy}")

    # SAE
    sae = SparseAutoencoder(d_in=d_model, n_features=cfg.n_features).to(cfg.device)
    opt = AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Save config
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    sae.train()
    pbar = tqdm(range(1, cfg.max_steps + 1), desc="SAE training")
    train_iter = batch_iter(train_data, cfg.batch_size, shuffle=True)

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = batch_iter(train_data, cfg.batch_size, shuffle=True)
            batch = next(train_iter)

        prompts = [x["prompt"] for x in batch]

        # Extract activations (fp32 for stable SAE training)
        x = get_layer_mlp_out_activations(
            model, prompts, cfg.layer, cfg.token_strategy, cfg.max_prompt_tokens
        ).float()  # [B, d_model]

        x_hat, h = sae(x)
        mse = F.mse_loss(x_hat, x)
        l1 = h.abs().mean()
        loss = mse + cfg.l1_coef * l1

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0:
            pbar.set_postfix({
                "loss": float(loss.item()),
                "mse": float(mse.item()),
                "l1": float(l1.item()),
            })

        if step % cfg.eval_every == 0:
            metrics = eval_recon(model, sae, val_data, cfg, n_batches=20)
            print(f"\n[Eval @ step {step}] mse={metrics['mse']:.6f}  mean|h|={metrics['mean_abs_h']:.6f}")
            save_checkpoint(cfg.out_dir, sae, cfg, step)

    # final save
    save_checkpoint(cfg.out_dir, sae, cfg, cfg.max_steps)
    print(f"Done. Checkpoints saved under: {cfg.out_dir}")

    # Optional: quick feature inspection demo
    # Pick a random feature and print its top prompts
    feat_id = random.randint(0, cfg.n_features - 1)
    top = feature_top_examples(model, sae, val_data, cfg, feature_id=feat_id, top_k=5, sample_n=2000)
    print(f"\n[Top prompts for feature {feat_id}]")
    for score, prompt in top:
        print(f"  {score:.4f} | {prompt[:120].replace('\\n',' ')}...")


if __name__ == "__main__":
    main()


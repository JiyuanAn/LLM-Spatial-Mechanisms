# Sparse Autoencoder (SAE) Training

这个目录包含用于训练稀疏自编码器(SAE)的代码，用于分析和理解 Transformer 模型的内部表示。

## 功能特性

- **多层提取**：可以从任意 Transformer 层提取 MLP 输出激活
- **灵活的 Token 策略**：支持多种 token 选择策略（last, question_mark, answer_prefix）
- **稀疏自编码器**：使用 L1 正则化训练稀疏表示
- **特征分析**：可以找到每个特征激活最高的样本

## 使用方法

### 1. 准备数据

支持两种数据格式：

**格式 1：JSONL（推荐用于大数据集）**
每行包含一个 JSON 对象，至少包含 `prompt` 字段：

```jsonl
{"prompt": "What is the capital of France?"}
{"prompt": "Explain quantum computing in simple terms."}
{"prompt": "How do neural networks work?"}
```

**格式 2：标准 JSON 数组**
整个文件是一个 JSON 数组，每个元素至少包含 `prompt`（或 `question`、`text`、`input`）字段：

```json
[
  {"prompt": "What is the capital of France?"},
  {"prompt": "Explain quantum computing in simple terms."},
  {"question": "How do neural networks work?"}
]
```

**支持的字段名**：`prompt`, `question`, `text`, `input`（自动映射为 `prompt`）

### 2. 运行训练

基本用法：

```bash
python train_sae.py \
    --model_path "Qwen/Qwen2.5-7B-Instruct" \
    --data_path data/prompts.jsonl \
    --out_dir outputs/sae_layer8 \
    --layer 8 \
    --n_features 4096 \
    --max_steps 5000
```

### 3. 参数说明

#### 必需参数
- `--model_path`: HuggingFace 模型路径或本地路径
- `--data_path`: JSONL 数据文件路径
- `--out_dir`: 输出目录，用于保存检查点和配置

#### 模型配置
- `--layer`: 要提取激活的层数（默认：8）
- `--token_strategy`: Token 选择策略
  - `last`: 使用最后一个非填充 token（默认）
  - `question_mark`: 使用问号 "?" 位置的 token
  - `answer_prefix`: 使用 "Answer:" 之前的 token

#### SAE 超参数
- `--n_features`: SAE 特征数量（默认：4096）
- `--l1_coef`: L1 正则化系数（默认：3e-4）
- `--lr`: 学习率（默认：2e-4）
- `--weight_decay`: 权重衰减（默认：0.0）
- `--grad_clip`: 梯度裁剪值（默认：1.0）

#### 训练配置
- `--batch_size`: 批次大小（默认：4）
- `--max_steps`: 最大训练步数（默认：5000）
- `--log_every`: 日志记录频率（默认：50）
- `--eval_every`: 评估和保存频率（默认：500）
- `--max_prompt_tokens`: 最大 prompt token 数（默认：2048）
- `--seed`: 随机种子（默认：42）

## 输出文件

训练完成后，输出目录将包含：

```
outputs/sae_layer8/
├── train_config.json       # 训练配置
├── sae_step_500.pt        # 检查点（每 eval_every 步保存）
├── sae_step_1000.pt
├── ...
└── sae_step_5000.pt       # 最终模型
```

## 检查点格式

每个检查点文件包含：
- `step`: 训练步数
- `cfg`: 完整的训练配置
- `state_dict`: SAE 模型的状态字典

加载检查点示例：

```python
import torch
from train_sae import SparseAutoencoder

# 加载检查点
ckpt = torch.load("outputs/sae_layer8/sae_step_5000.pt")
cfg = ckpt["cfg"]

# 重建模型
sae = SparseAutoencoder(d_in=cfg["d_in"], n_features=cfg["n_features"])
sae.load_state_dict(ckpt["state_dict"])
sae.eval()
```

## 注意事项

1. **数据大小**：建议至少提供数百到数千个样本用于训练
2. **内存需求**：根据模型大小和批次大小调整，建议在 GPU 上运行
3. **超参数调整**：
   - `l1_coef` 控制稀疏性：值越大，激活越稀疏
   - `n_features` 通常设置为 `d_model` 的 1-16 倍
   - 较小的 `lr` 和 `batch_size` 可能需要更多训练步数

## 示例：完整训练流程

```bash
# 1. 准备数据
cat > data/my_prompts.jsonl << EOF
{"prompt": "Translate to French: Hello, how are you?"}
{"prompt": "What is 2+2?"}
{"prompt": "Write a poem about AI"}
EOF

# 2. 训练 SAE（层 16，8192 特征）
python train_sae.py \
    --model_path "Qwen/Qwen2.5-7B-Instruct" \
    --data_path data/my_prompts.jsonl \
    --out_dir outputs/sae_layer16_8k \
    --layer 16 \
    --n_features 8192 \
    --l1_coef 2e-4 \
    --batch_size 8 \
    --max_steps 10000 \
    --token_strategy last

# 3. 查看结果
ls -lh outputs/sae_layer16_8k/
```

## 特征分析

训练完成后，脚本会自动展示一个随机特征的 top-k 激活样本。您可以使用 `feature_top_examples` 函数来分析特定特征：

```python
from train_sae import feature_top_examples, TrainConfig, SparseAutoencoder
from transformer_lens import HookedTransformer

# 加载模型和 SAE
model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# ... 加载 SAE 检查点 ...

# 分析特征 42
top_prompts = feature_top_examples(
    model, sae, val_data, cfg, 
    feature_id=42, top_k=10
)

for score, prompt in top_prompts:
    print(f"{score:.4f} | {prompt}")
```


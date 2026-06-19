# Transformer 2.3B

A from-scratch implementation of a modern, decoder-only (GPT-style) Transformer language model in **PyTorch**. The architecture follows the design choices popularized by LLaMA / Mistral-class models, including **RoPE positional embeddings**, **Grouped/Multi-Query Attention (GQA/MQA)**, **RMSNorm**, **SwiGLU** feed-forward layers, and **parallel residual** blocks.

The default configuration describes a ~2.3B-parameter model, but the training scripts ship with a tiny config so you can train, generate, and inspect the full pipeline end-to-end on a single CPU or modest GPU.

---

## ✨ Features

- **Decoder-only Transformer** built from scratch (no `nn.Transformer`).
- **Rotary Positional Embeddings (RoPE)** — relative position encoding applied to queries and keys.
- **Grouped / Multi-Query Attention** — fewer KV heads than query heads for memory-efficient attention.
- **RMSNorm** — normalization without mean subtraction.
- **SwiGLU** feed-forward network — gated activation for stronger MLP capacity.
- **Parallel residual blocks** — attention and MLP computed from the same normalized input (configurable).
- **Custom BPE tokenizer** (200k vocab) trainable via 🤗 `tokenizers`, with a character-level fallback.
- **Training utilities** — AdamW optimizer, cosine LR schedule, causal dataset, and a `Trainer` with train/eval loops.
- **Performance tracking & plotting** — metrics saved to `performance.json` and rendered to `performance.png`.

---

## 🏗️ Architecture

| Component | Implementation | File |
|---|---|---|
| Model wrapper | Token embedding → N blocks → norm → LM head | `model_f/model.py` |
| Transformer block | RMSNorm + Attention + SwiGLU (parallel residual) | `model_f/block.py` |
| Attention | Multi/Grouped-Query Attention with RoPE | `model_f/attention.py` |
| Feed-forward | SwiGLU (`w3(silu(w1·x) * w2·x)`) | `model_f/feedforwaord.py` |
| Positional encoding | RoPE sin/cos cache + apply | `model_f/rope.py` |
| Normalization | RMSNorm | `model_f/rmsmorm.py` |
| Hyperparameters | `Config` dataclass-style container | `model_f/config.py` |

### Default model config (`model_f/config.py`)

| Hyperparameter | Value |
|---|---|
| `vocab_size` | 200,000 |
| `max_seq_len` | 4,096 |
| `dim` | 4,096 |
| `n_layers` | 48 |
| `n_heads` | 32 |
| `kv_heads` | 4 |
| `dim_ff` | 14,336 |
| `rope_theta` | 10,000 |
| `dropout` | 0.0 |
| `bias` | False |
| `parallel_residual` | True |

> **Note:** The training scripts (`train.py` / `train_v2.py`) override these with a much smaller config (e.g. `dim=128`, `n_layers=1`) so the full pipeline runs quickly on CPU. Adjust as your hardware allows.

---

## 📁 Project structure

```
transformer_2.3B/
├── model_f/                  # Model architecture
│   ├── config.py             # Hyperparameter config
│   ├── model.py              # Top-level Model (embeddings, blocks, head)
│   ├── block.py              # TransformerBlock (parallel residual)
│   ├── attention.py          # Multi/Grouped-Query Attention
│   ├── feedforwaord.py       # SwiGLU MLP
│   ├── rope.py               # Rotary positional embeddings
│   └── rmsmorm.py            # RMSNorm
│
├── training/                 # Training building blocks
│   ├── dataset.py            # CausalTextDataset (next-token targets)
│   ├── optimizer.py          # AdamW factory
│   ├── schedules.py          # Cosine LR schedule
│   └── trainer.py            # Trainer: train_epoch / evaluate
│
├── scripts/                  # Entry points
│   ├── train.py              # Minimal training loop
│   ├── train_v2.py           # Training w/ validation, metrics, fallbacks
│   ├── generate.py           # Text generation from a checkpoint
│   └── plot_performance.py   # Plot metrics from performance.json
│
├── tokenizer/
│   ├── build_tokenizer.py    # Train a BPE tokenizer
│   └── jarvis_tokenizer.json # Saved tokenizer
│
├── data/                     # CSV datasets (train/validation/test)
│   ├── pile.py               # Download a HF dataset to CSV
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── performance.json          # Latest training metrics
└── model_epoch_1.pth         # Saved checkpoint
```

---

## 🚀 Getting started

### 1. Install dependencies

```bash
pip install torch tokenizers datasets pandas tqdm matplotlib
```

### 2. (Optional) Build a tokenizer

`scripts/*` expect a tokenizer at `tokenizer/jarvis_tokenizer.json`. One is already included, but you can rebuild it:

```bash
python tokenizer/build_tokenizer.py
```

> The BPE trainer reads a CSV with a `dialog` column. Update the path/column inside `build_tokenizer.py` to match your data. If the BPE tokenizer can't load, `train_v2.py` automatically falls back to a character-level tokenizer.

### 3. Prepare data

Training reads from `data/train.csv` (and optionally `data/validation.csv`), using the **`dialog`** column by default. To pull a large public corpus into CSV form:

```bash
python data/pile.py     # downloads HuggingFaceFW/fineweb-edu -> CSV
```

### 4. Train

Minimal run:

```bash
python scripts/train.py
```

Recommended run with validation + metrics:

```bash
python scripts/train_v2.py
```

`train_v2.py` is configurable via environment variables:

| Env var | Default | Purpose |
|---|---|---|
| `EPOCHS` | `1` | Number of training epochs |
| `MAX_TRAIN_SAMPLES` | `2000` | Cap training samples (`0` = use all) |
| `MAX_VAL_SAMPLES` | `500` | Cap validation samples (`0` = use all) |

Example:

```bash
EPOCHS=3 MAX_TRAIN_SAMPLES=0 python scripts/train_v2.py
```

Checkpoints are written to `model_epoch_<N>.pth` and metrics to `performance.json`.

### 5. Generate text

```bash
python scripts/generate.py
```

> `generate.py` loads a checkpoint named `model.pth` at the project root. Rename/copy a saved checkpoint (e.g. `cp model_epoch_1.pth model.pth`) or edit the `model_path` in the script first. Make sure the `Config` used for generation matches the one used during training.

### 6. Plot performance

```bash
python scripts/plot_performance.py   # writes performance.png
```

---

## 📊 Training pipeline

1. **Load text** from a CSV `dialog` column.
2. **Tokenize** with the BPE tokenizer (or character-level fallback).
3. **Build a `CausalTextDataset`** that yields `(x, y)` where `y` is `x` shifted by one token.
4. **Train** with AdamW + cosine LR schedule, minimizing cross-entropy next-token loss.
5. **Evaluate** on the validation set (loss + perplexity).
6. **Save** checkpoints, `performance.json`, and a `performance.png` plot.

---

## ⚙️ Notes & caveats

- The default `Config` is large (~billions of params). Use the reduced config in the training scripts unless you have substantial GPU memory.
- Generation uses temperature-based sampling and recomputes the full forward pass each step (no KV cache yet) — fine for experimentation, slow for long sequences.
- `RMSNorm` here normalizes by the full L2 norm (not the standard root-mean-square); adjust if you want exact LLaMA semantics.
- The top-level `model.norm` is a `Linear` layer rather than a final RMSNorm — swap it if you want a more conventional final-norm setup.

---

## 📜 License

No license file is currently included. Add one (e.g. MIT, Apache-2.0) if you intend to share or open-source this project.

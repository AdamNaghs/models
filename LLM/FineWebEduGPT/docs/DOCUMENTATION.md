# FineWebEduGPT -- Technical Documentation

A GPT language model trained from scratch on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), then instruction-tuned for multi-turn chat.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [File Reference](#3-file-reference)
4. [Dependencies](#4-dependencies)
5. [Pretraining (`train_fineweb_gpt.py`)](#5-pretraining-train_fineweb_gptpy)
6. [Size-Based Presets](#6-size-based-presets)
7. [Data Loading Modes](#7-data-loading-modes)
8. [Tokenizer](#8-tokenizer)
9. [Checkpoints](#9-checkpoints)
10. [Distributed Training](#10-distributed-training)
11. [SLURM Deployment](#11-slurm-deployment)
12. [Chat Finetuning (`finetune_chat.py`)](#12-chat-finetuning-finetune_chatpy)
13. [Inference (`chat_fineweb_gpt.py`)](#13-inference-chat_fineweb_gptpy)
14. [Hyperparameter Reference](#14-hyperparameter-reference)
15. [Design Decisions](#15-design-decisions)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Project Overview

FineWebEduGPT is a two-stage training pipeline:

```
Stage 1: Pretraining                    Stage 2: SFT Finetuning
┌─────────────────────────┐            ┌─────────────────────────┐
│  FineWeb-Edu (web text) │            │  UltraChat 200k (chat)  │
│  Next-token prediction  │──ckpt───>  │  Masked loss (asst only)│
│  train_fineweb_gpt.py   │            │  finetune_chat.py       │
└─────────────────────────┘            └─────────────────────────┘
         │                                        │
    fineweb_gpt.ckpt                     fineweb_gpt_chat.ckpt
                                                  │
                                         chat_fineweb_gpt.py
```

### Stage Intent

- **Stage 1 (pretraining):** build general language competence, factual grounding, and reasoning patterns from broad educational web text.
- **Stage 2 (SFT):** shape behavior for instruction following and conversational turn-taking.

### Typical Workflow

```bash
OUT_DIR=runs/350m

# 1) Pretrain (single GPU)
python train_fineweb_gpt.py -350M --out-dir "$OUT_DIR"

# 2) Finetune
python finetune_chat.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt"

# 3) Chat
python chat_fineweb_gpt.py --ckpt "$OUT_DIR/fineweb_gpt_chat.ckpt"
```

---

## 2. Architecture

The model is a decoder-only Transformer (GPT-style), pre-norm, with weight tying.

### Block Structure

```
Input
  │
  ├──── LayerNorm ──> QKV Linear ──> Multi-Head Attention ──> Output Projection ──> Dropout
  │                                                                                    │
  │<──────────────────────── Residual Connection ─────────────────────────────────────+
  │
  ├──── LayerNorm ──> Linear(4x) ──> GELU ──> Linear(1x) ──> Dropout
  │                                                              │
  │<──────────────────────── Residual Connection ───────────────+
  │
Output
```

### Components

| Component | Implementation | Notes |
|-----------|---------------|-------|
| Attention | `F.scaled_dot_product_attention` | Flash/memory-efficient attention on CUDA, causal mask fallback on CPU |
| QKV Projection | Single fused `nn.Linear(n_embd, 3*n_embd)` | Bias disabled |
| Output Projection | `nn.Linear(n_embd, n_embd)` | Bias disabled |
| FFN | `Linear(E, 4E) -> GELU -> Linear(4E, E)` | Standard 4x expansion |
| Normalization | `nn.LayerNorm` | Pre-norm |
| Positional Encoding | `nn.Embedding(context, n_embd)` | Learned absolute positions |
| Weight Tying | `head.weight = tok.weight` | Shared token/input-output representation |

### Parameter Estimate

Approximate total:

```
params = vocab * E + context * E + n_layer * (12 * E^2 + 9 * E)
```

where `E = n_embd`.

---

## 3. File Reference

```
FineWebEduGPT/
├── train_fineweb_gpt.py       # Stage 1 pretraining
├── finetune_chat.py           # Stage 2 SFT
├── chat_fineweb_gpt.py        # Inference chat loop
├── fineweb_gpt_common.py      # Shared model/chat/artifact helpers
├── smoke_test.py              # Fast local sanity checks
├── launch_torchrun.sh         # Multi-GPU launcher by preset
├── star_gpu7_fineweb.sbatch   # Example SLURM pipeline script
├── README.md                  # Quick start
├── docs/
│   └── DOCUMENTATION.md       # This document
│
│ # Runtime outputs (under OUT_DIR, default: runs/<preset>/):
├── tokenizer.model
├── tokenizer.vocab
├── tokenizer_seed.txt
├── fineweb_gpt.ckpt
├── fineweb_gpt_chat.ckpt
└── fineweb_gpt_chat_stepN.ckpt
```

---

## 4. Dependencies

Required:

```
torch >= 2.1
datasets >= 2.14
sentencepiece >= 0.1.99
numpy
```

Common optional/runtime extras:

```
pyarrow
huggingface_hub
```

Install:

```bash
pip install -r requirements.txt
python smoke_test.py
```

---

## 5. Pretraining (`train_fineweb_gpt.py`)

### Purpose

Train with standard causal LM objective (next-token prediction).

### Common Commands

```bash
OUT_DIR=runs/350m

# Default run (single GPU, no shorthand preset)
python train_fineweb_gpt.py --out-dir runs/custom

# Recommended 350M preset
python train_fineweb_gpt.py -350M --out-dir "$OUT_DIR"

# Alternate preset and crawl config
python train_fineweb_gpt.py -760M --config CC-MAIN-2025-26 --out-dir runs/760m

# Resume
python train_fineweb_gpt.py -350M --out-dir "$OUT_DIR" --resume "$OUT_DIR/fineweb_gpt.ckpt"
```

### Training Loop (high level)

1. Build/load tokenizer.
2. Build batchers (train + validation).
3. Forward pass under autocast (CUDA).
4. Gradient accumulation + optional `no_sync()` for non-final micro-steps in DDP.
5. Clip gradients to max norm 1.0.
6. Warmup + cosine LR schedule.
7. Periodic eval (loss, perplexity).
8. Atomic checkpointing.

### Graceful Shutdown

`SIGTERM`/`SIGINT` handlers set a shutdown flag, allow loaders to drain, save checkpoint, and exit cleanly.

---

## 6. Size-Based Presets

Presets are now **model-size oriented**, not GPU-brand oriented.

### Supported preset flags

- `-125M`  -> `--preset 125m`
- `-350M`  -> `--preset 350m`
- `-760M`  -> `--preset 760m`
- `-1.3B`  -> `--preset 1.3b`

### Preset table

| Preset | Approx Params | Layers | Heads | Embedding | Context | Batch Size | Grad Accum | Peak LR | Min LR | Warmup | Train Steps |
|--------|----------------|--------|-------|-----------|---------|------------|------------|---------|--------|--------|------------|
| `125m` | ~125M | 12 | 12 | 768 | 2048 | 64 | 2 | 6e-4 | 6e-5 | 1000 | 20,000 |
| `350m` | ~356M | 24 | 16 | 1024 | 2048 | 32 | 4 | 3e-4 | 3e-5 | 2000 | 30,000 |
| `760m` | ~760M | 24 | 16 | 1536 | 2048 | 16 | 8 | 2.5e-4 | 2.5e-5 | 2000 | 40,000 |
| `1.3b` | ~1.3B | 29 | 16 | 1856 | 2048 | 8 | 16 | 2e-4 | 2e-5 | 3000 | 50,000 |

### Override semantics

Preset values are applied unless explicitly overridden by CLI flags.

Example:

```bash
python train_fineweb_gpt.py -350M --batch-size 16 --lr 2.5e-4
```

This keeps all `350m` defaults except the explicitly overridden fields.

---

## 7. Data Loading Modes

### A) Local full download (default)

```bash
python train_fineweb_gpt.py -350M
```

- Uses HuggingFace local cache.
- Best throughput once cached.
- Highest local disk usage.

### B) Streaming mode

```bash
python train_fineweb_gpt.py -350M --stream
```

- Reads directly from Hub.
- Minimal disk requirements.
- Slower and network dependent.

### C) Rolling cache mode

```bash
python train_fineweb_gpt.py -350M --cache-gb 5
```

- Downloads a chunk of parquet data up to `N` GB.
- Trains through it, rotates chunk, continues.
- Good middle ground between storage and throughput.

### Train/Val split behavior

For non-rolling local mode, the loader performs a held-out split:
- **Train**: first 99%
- **Val**: last 1% (minimum 100 documents)

This avoids train/val overlap by construction.

---

## 8. Tokenizer

SentencePiece BPE is used for pretraining, finetuning, and inference.

### Auto-build flow

If `<out_dir>/tokenizer.model` is missing:
1. Stream `--seed-docs` documents from current FineWeb config.
2. Write text to `tokenizer_seed.txt`.
3. Train SentencePiece with configured `--vocab-size`.
4. Save `tokenizer.model` and `tokenizer.vocab` under `OUT_DIR`.

### Special token IDs

| Token | ID |
|-------|----|
| `<unk>` | 0 |
| `<s>` | 1 |
| `</s>` | 2 |
| `<pad>` | 3 |

### DDP behavior

Rank 0 builds tokenizer, then `dist.barrier()` gates other ranks before load.

---

## 9. Checkpoints

### Pretraining checkpoint keys

- `state_dict`
- `opt_state_dict`
- `scaler_state_dict`
- `args`
- `vocab`
- `step`
- `cache_next_shard_idx` (when rolling cache is active)
- tokenizer fingerprint metadata (used for resume safety)

### Finetuning checkpoint keys

- `state_dict`
- `args` (base pretrain args)
- `finetune_args`
- `chat_format`
- `best_val_loss`
- `step`, `epoch`, `vocab`

### Atomic write guarantee

All checkpoint writes are: `tmp file -> os.replace(tmp, final)`.

---

## 10. Distributed Training

### Backend selection

- CUDA -> NCCL
- CPU-only -> Gloo

### DDP details

- One process per GPU via `torchrun`.
- Rank-local seeds (`seed + rank`) for deterministic but distinct data order.
- Accumulation steps use `no_sync()` except on the final micro-batch.

### Launch examples

```bash
# Direct torchrun
torchrun --standalone --nproc_per_node=8 train_fineweb_gpt.py --preset 350m --out-dir runs/350m

# Convenience wrapper
./launch_torchrun.sh 350m 8
./launch_torchrun.sh 1.3b 8 /scratch/$USER/fineweb-1.3b
```

---

## 11. SLURM Deployment

`star_gpu7_fineweb.sbatch` runs pretrain -> checkpoint check -> SFT.

### Submit

```bash
sbatch star_gpu7_fineweb.sbatch
```

### Current stage-1 invocation

```bash
OUT_DIR=${OUT_DIR:-runs/350m}

torchrun --standalone --nproc_per_node=8 train_fineweb_gpt.py \
  --preset 350m \
  --out-dir "$OUT_DIR" \
  --cache-gb 500 \
  --num-workers 8 \
  --queue-size 25
```

### Notes

- Script installs requirements at runtime.
- It verifies `$OUT_DIR/fineweb_gpt.ckpt` exists before starting SFT.
- Adjust `#SBATCH` directives for your cluster constraints.

---

## 12. Chat Finetuning (`finetune_chat.py`)

### Purpose

Convert pretrained next-token model into instruction/chat model with assistant-only supervised loss.

### Usage

```bash
OUT_DIR=runs/350m
python finetune_chat.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt"

# quick run
python finetune_chat.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --max-samples 5000
```

### Loss masking

Only assistant tokens contribute to objective:

```python
per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
loss = (per_token_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
```

This prevents wasting capacity on predicting user text.

---

## 13. Inference (`chat_fineweb_gpt.py`)

### Usage

```bash
OUT_DIR=runs/350m
python chat_fineweb_gpt.py --ckpt "$OUT_DIR/fineweb_gpt_chat.ckpt"
python chat_fineweb_gpt.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --raw
```

### Sampling

Nucleus sampling (`top-p`) with temperature scaling.

Stops on:
- EOS token
- turn boundary markers
- max token limit

---

## 14. Hyperparameter Reference

### Pretraining (`train_fineweb_gpt.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `CC-MAIN-2025-26` | FineWeb-Edu crawl snapshot |
| `--train-steps` | 100000 | Total optimizer steps |
| `--batch-size` | 8 | Per-GPU micro-batch |
| `--context` | 512 | Sequence length |
| `--n-layer` | 12 | Transformer layers |
| `--n-head` | 10 | Attention heads |
| `--n-embd` | 640 | Embedding width |
| `--dropout` | 0.1 | Dropout |
| `--lr` | 3e-4 | Peak LR |
| `--min-lr` | 3e-5 | Cosine floor |
| `--weight-decay` | 0.1 | AdamW weight decay |
| `--warmup-steps` | 1000 | Warmup steps |
| `--grad-accum` | 16 | Gradient accumulation |
| `--vocab-size` | 16000 | SentencePiece vocab size |
| `--eval-every` | 500 | Eval frequency |
| `--eval-iters` | 100 | Eval batches |
| `--ckpt-every` | 2000 | Checkpoint cadence |
| `--log-every` | 10 | Train log cadence |
| `--queue-size` | 64 | Data queue depth |
| `--num-workers` | 2 | Loader workers |
| `--seed` | 42 | Random seed |
| `--preset` | none | `125m`, `350m`, `760m`, `1.3b` |
| `--stream` | false | Stream from HF |
| `--cache-gb` | 0 | Rolling cache size |
| `--cache-dir` | `.data_cache` | Rolling cache directory |
| `--resume` | none | Resume checkpoint path |

### Finetuning (`finetune_chat.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt` | required | Pretrained checkpoint |
| `--tok` | `<ckpt_dir>/tokenizer.model` | Tokenizer path |
| `--output` | `<ckpt_dir>/fineweb_gpt_chat.ckpt` | Output checkpoint |
| `--dataset` | `ultrachat` | Dataset source |
| `--data-path` | none | Custom JSONL path |
| `--max-samples` | none | Data cap for quick tests |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--lr` | 2e-5 | Peak LR |
| `--min-lr` | 1e-6 | LR floor |
| `--weight-decay` | 0.01 | Weight decay |
| `--warmup-ratio` | 0.03 | Warmup fraction |
| `--grad-accum` | 8 | Gradient accumulation |
| `--dropout` | 0.05 | Finetune dropout |
| `--max-grad-norm` | 1.0 | Clip threshold |
| `--eval-every` | 200 | Eval cadence |
| `--eval-iters` | 50 | Eval batches |
| `--log-every` | 10 | Log cadence |
| `--ckpt-every` | 500 | Checkpoint cadence |
| `--seed` | 42 | Random seed |

---

## 15. Design Decisions

### Why size-based presets?

Hardware labels age badly and encourage brittle assumptions. Size presets focus on model objective first, then let operators tune runtime knobs (batch, accumulation, workers) for actual hardware.

### Why tokenizer fingerprint on resume?

Resuming with a different tokenizer silently corrupts token semantics. Fingerprinting prevents that class of failure.

### Why held-out split in local mode?

It guarantees train/val disjointness without depending on an external split definition.

### Why masked SFT loss?

Assistant-only supervision concentrates gradient signal on response quality and avoids learning to imitate user prompts.

---

## 16. Troubleshooting

### `ModuleNotFoundError` (for example `pyarrow`)

Install missing dependencies in your active env:

```bash
pip install -r requirements.txt
```

### CUDA OOM

Lower `--batch-size` or `--context`, then increase `--grad-accum` to preserve effective global batch.

### `torch.compile` failure

Non-fatal. Training continues without compile optimization.

### Rolling cache has no parquet files

Check network/HF access and ensure `huggingface_hub` is installed.

### Resume fails with tokenizer mismatch

Use the exact `tokenizer.model` from the original pretraining run.

### DDP timeout during long I/O windows

Large rolling-cache downloads can stall ranks. Increase timeout budget and verify node interconnect/network health.

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
17. [Evaluation](#17-evaluation)

---

For the exact Star login-node and Slurm operator workflow, use [`STAR_HPC.md`](./STAR_HPC.md).

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
├── run_eval.sh                # Quick eval wrapper
├── eval/
│   ├── model_adapter.py       # Checkpoint wrapper for scoring/generation/ppl
│   ├── benchmark_loaders.py   # HF benchmark loaders -> normalized rows
│   ├── eval_lm.py             # Held-out perplexity/avg NLL
│   ├── eval_mcq.py            # Multiple-choice benchmark runner
│   ├── eval_chat.py           # Prompt-suite chat regression runner
│   ├── contamination_scan.py  # Exact + n-gram contamination scan
│   ├── metrics.py             # Eval aggregates
│   └── utils.py               # JSONL/text helpers
├── eval_data/
│   └── chat_eval_prompts.jsonl # Fixed chat regression prompt suite
├── launch_torchrun.sh         # Multi-GPU launcher by preset
├── star_gpu7_fineweb.sbatch   # Example SLURM pipeline script
├── README.md                  # Quick start
├── docs/
│   ├── DOCUMENTATION.md       # This document
│   └── EVAL.md                # Eval workflow + contamination guidance
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

# Alternate preset and sample config
python train_fineweb_gpt.py -760M --config sample-100BT --out-dir runs/760m

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
| `125m` | ~125M | 12 | 12 | 768 | 2048 | 8 | 16 | 6e-4 | 6e-5 | 1000 | 20,000 |
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

### B) Offline staged chunk mode

```bash
python train_fineweb_gpt.py -125M \
  --offline \
  --local-data-dir /fs1/proj/educational_web_data/dataset/fineweb-edu/sample-10BT/source \
  --out-dir /fs1/proj/educational_web_data/runs/125m \
  --stop-after-one-epoch
```

- Reads only local parquet files from the staged chunk directory.
- Performs no HuggingFace network access when `--offline` is set.
- Stops after one full pass over the staged chunk so operators can stage the next one manually.
- The preferred Star workflow is one staged sample config at a time.

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
1. If `--local-data-dir` is set, read `--seed-docs` documents from staged local parquet.
2. Otherwise load `--seed-docs` documents from the selected FineWeb-Edu config.
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

### Offline failure behavior

If `--offline` is set:
- missing staged parquet under `--local-data-dir` is a hard error
- missing tokenizer falls back only to staged local parquet, never to HuggingFace

---

## 9. Star HPC Manual Chunk Workflow

### Goal

Use the login node for downloads, and the compute node only for training.

### Step 1: Stage the next sample chunk on the login node

```bash
cd LLM/FineWebEduGPT
python download_fineweb_snapshot.py \
  --config sample-100BT \
  --max-gb 500
```

This script:
- stages one `source/` directory for the requested sample
- downloads the next shard window into `/fs1/proj/educational_web_data/dataset/fineweb-edu/<config>/source`
- writes `_chunk_manifest.json` in the source directory
- updates `.download_state.json` under `/fs1/proj/educational_web_data/dataset/fineweb-edu/<config>/`

Run it again after each training job to stage the next chunk for that sample.

### Step 2: Submit training

```bash
LOCAL_DATA_DIRS=/fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source
sbatch --qos=long2x --export=ALL,OUT_DIR=/fs1/proj/educational_web_data/runs/1.3b,LOCAL_DATA_DIRS="$LOCAL_DATA_DIRS",CONFIGS=sample-100BT \
  -o /fs1/proj/educational_web_data/logs/fineweb-1-3b-%j.out \
  -e /fs1/proj/educational_web_data/logs/fineweb-1-3b-%j.err \
  star_gpu7_fineweb_1_3b.sbatch
```

The Star sbatch files:
- use `--offline` plus one `--local-data-dir` for the staged sample
- resume from `<out_dir>/fineweb_gpt.ckpt` if it already exists
- stop after one epoch over the staged sample chunk
- do not launch chat finetuning automatically
- auto-clean the disposable local dataset cache unless `KEEP_LOCAL_DATASET_CACHE=1`

### Step 3: Repeat

After the job finishes:
1. run `download_fineweb_snapshot.py` again
2. resubmit the same sbatch file
3. keep the same `OUT_DIR` so pretraining resumes from the latest checkpoint

---

## 10. Checkpoints

### Pretraining checkpoint keys

- `state_dict`
- `opt_state_dict`
- `scaler_state_dict`
- `args`
- `vocab`
- `step`
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

## 11. Distributed Training

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

Use [`STAR_HPC.md`](./STAR_HPC.md) for the exact Star operator workflow.

Current Star defaults:
- `star_gpu7_fineweb_125m.sbatch` uses H100-safe `125m` settings: `BATCH_SIZE=8`, `GRAD_ACCUM=16`, `--no-compile`
- `star_gpu7_fineweb_1_3b.sbatch` uses H100-safe `1.3b` settings: `BATCH_SIZE=1`, `GRAD_ACCUM=128`, `NO_COMPILE=1`
- both support offline multi-config training via `CONFIGS` and `LOCAL_DATA_DIRS`

### Submit

```bash
sbatch --qos=long2x \
  -o /fs1/proj/educational_web_data/logs/fineweb-1-3b-%j.out \
  -e /fs1/proj/educational_web_data/logs/fineweb-1-3b-%j.err \
  star_gpu7_fineweb_1_3b.sbatch
```

### Safe `1.3b` smoke run

```bash
LOCAL_DATA_DIRS=/fs1/proj/educational_web_data/dataset/fineweb-edu/sample-10BT/source
sbatch --qos=long2x \
  --export=ALL,OUT_DIR=/fs1/proj/educational_web_data/runs/1.3b-smoke,LOCAL_DATA_DIRS="$LOCAL_DATA_DIRS",CONFIGS=sample-10BT,BATCH_SIZE=1,GRAD_ACCUM=32,NO_COMPILE=1,TRAIN_STEPS=20,EVAL_EVERY=10,EVAL_ITERS=2,CKPT_EVERY=20 \
  -o /fs1/proj/educational_web_data/logs/fineweb-1-3b-smoke-%j.out \
  -e /fs1/proj/educational_web_data/logs/fineweb-1-3b-smoke-%j.err \
  star_gpu7_fineweb_1_3b.sbatch
```

### Notes

- The `1.3b` sbatch trains offline over `sample-100BT` by default.
- It resumes from `/fs1/proj/educational_web_data/runs/1.3b/fineweb_gpt.ckpt` when present.
- It stops after one full pass over the staged sample chunk if it has not yet reached `train_steps`.
- Use `TRAIN_STEPS`, `EVAL_EVERY`, `EVAL_ITERS`, and `CKPT_EVERY` as env overrides for smoke runs.
- The derived local dataset cache is disposable and auto-cleaned by default.

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
| `--config` | `sample-100BT` | FineWeb-Edu dataset config |
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

Large staged dataset indexing or local cache materialization can stall ranks. Increase timeout budget and verify node interconnect/network health.

---

## 17. Evaluation

The repo includes a lightweight evaluation stack that works directly against the native checkpoint format. It is designed to answer four questions quickly:

1. Did the base model actually learn language?
2. How does it perform on small but meaningful reasoning benchmarks?
3. Does the chat-tuned model behave coherently and follow instructions?
4. Is benchmark performance inflated by contamination?

### Evaluation entry points

```bash
OUT_DIR=runs/350m

# Held-out LM sanity check
python eval/eval_lm.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --dataset wikitext_valid

# Multiple-choice benchmarks
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench hellaswag
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench piqa
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench winogrande
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench arc_challenge

# Chat regression suite
python eval/eval_chat.py --ckpt "$OUT_DIR/fineweb_gpt_chat.ckpt" --prompts eval_data/chat_eval_prompts.jsonl

# Convenience wrapper (derives fineweb_gpt.ckpt from the same OUT_DIR)
bash run_eval.sh "$OUT_DIR/fineweb_gpt_chat.ckpt"
```

### Included benchmark coverage

- `eval_lm.py`: held-out perplexity / average negative log-likelihood on external text sets (`wikitext_valid`, `wikitext_test`, `lambada`)
- `eval_mcq.py`: HellaSwag, PIQA, Winogrande, ARC-Easy, ARC-Challenge
- `eval_chat.py`: fixed prompt suite covering factual QA, summarization, reasoning, safety, formatting, coding-lite prompts, and multi-turn memory
- `contamination_scan.py`: exact-match + n-gram-overlap audit over local corpora and/or sampled FineWeb-Edu docs

### Multiple-choice scoring method

`eval_mcq.py` uses teacher-forced choice scoring:

1. tokenize `prompt + choice`
2. run the model once under teacher forcing
3. sum log-probabilities only over the choice tokens
4. select the choice with the highest score

By default, selection uses **length-normalized log-probability** to reduce bias toward short answers. Use `--metric raw` if you want raw total log-probability instead.

### Contamination scanning workflow

```bash
python eval/contamination_scan.py \
  --bench hellaswag \
  --bench piqa \
  --fineweb-config sample-100BT \
  --fineweb-sample-docs 5000 \
  --ckpt-label fineweb-scan
```

Outputs:

- `benchmark_manifest.jsonl`
- `contamination_findings.jsonl`
- `contamination_summary.json`

Status meanings:

- `clean`: no strong overlap found in the scanned source set
- `suspected`: exact prompt overlap or high n-gram overlap
- `contaminated`: exact prompt + gold-answer overlap found

### Recommended usage order

1. `eval_lm.py`
2. `eval_mcq.py`
3. `eval_chat.py`
4. `contamination_scan.py`

That order matters. If the base model is weak, contamination analysis is mostly academic.

### Additional reference

See `docs/EVAL.md` for the full operator guide, example commands, and interpretation notes.

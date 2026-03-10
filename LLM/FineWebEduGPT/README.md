# FineWebEduGPT

A GPT language model trained from scratch on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), with a supervised finetuning (SFT) pipeline for turning it into a chatbot.

## Architecture

| Component | Detail |
|-----------|--------|
| Attention | Multi-head with `scaled_dot_product_attention`, causal masking |
| FFN | GELU activation, 4x expansion |
| Normalization | Pre-LayerNorm (norm before attention and FFN) |
| Weight tying | Embedding and output projection share weights |
| Positional encoding | Learned absolute position embeddings |

## Pipeline

The project has two training stages:

### Stage 1: Pretraining (`train_fineweb_gpt.py`)

Next-token prediction on raw web text from FineWeb-Edu. This teaches the model language, facts, and reasoning patterns.

### Stage 2: Chat Finetuning (`finetune_chat.py`)

Supervised finetuning on multi-turn conversations from [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). This teaches the model to follow instructions and hold conversations.

Key differences from pretraining:
- **Masked loss**: only assistant responses contribute to the loss. User messages and formatting tokens are masked out, so the model learns *what to say* without being penalized for not predicting the user's input.
- **Lower learning rate** (2e-5 vs 3e-4) to avoid catastrophic forgetting of pretrained knowledge.
- **Chat format**: conversations are structured as `### User:\n{message}\n### Assistant:\n{response}\n</s>` using regular subword tokens -- no vocabulary modifications needed.

### Stage 3: Chat (`chat_fineweb_gpt.py`)

Interactive inference with multi-turn history, top-p sampling, and automatic stop on EOS or turn boundaries.

## Setup

```bash
pip install -r requirements.txt
python smoke_test.py
```

By default, pretraining writes artifacts to `/fs1/proj/educational_web_data/runs/<preset>/`, and staged local parquet lives under `/fs1/proj/educational_web_data/dataset/fineweb-edu/<config>/source/`. Override the root with `FINEWEB_STORAGE_ROOT` or pass explicit `--out-dir`. Finetuning and chat can auto-discover `tokenizer.model` from the checkpoint directory.

Evaluation docs live in [`docs/EVAL.md`](docs/EVAL.md).
Star HPC usage docs live in [`docs/STAR_HPC.md`](docs/STAR_HPC.md).

## Pretraining

Default run (single GPU, 350M preset):

```bash
python train_fineweb_gpt.py -350M
```

Pick a FineWeb-Edu sample config:

```bash
python train_fineweb_gpt.py -350M --config sample-100BT
```

### Size-Based Presets (multi-GPU tuned defaults)

| Flag | Parameters | Layers | Heads | Embed | Context | Batch | Grad Accum | LR | Steps |
|------|-----------|--------|-------|-------|---------|-------|------------|-----|-------|
| `-125M` | ~125M | 12 | 12 | 768 | 2048 | 8 | 16 | 6e-4 | 20k |
| `-350M` | ~356M | 24 | 16 | 1024 | 2048 | 32 | 4 | 3e-4 | 30k |
| `-760M` | ~760M | 24 | 16 | 1536 | 2048 | 16 | 8 | 2.5e-4 | 40k |
| `-1.3B` | ~1.3B | 29 | 16 | 1856 | 2048 | 8 | 16 | 2e-4 | 50k |

Learning rates follow GPT-3/Chinchilla scaling (smaller LR for larger models).
Warmup: 1000 steps (125M), 2000 (350M/760M), 3000 (1.3B).

```bash
python train_fineweb_gpt.py -125M
python train_fineweb_gpt.py -350M
python train_fineweb_gpt.py -1.3B
```

### Data Loading Modes

- **Default**: downloads the selected FineWeb-Edu config to the HuggingFace local cache, then trains from memory-mapped Arrow files. Fastest throughput on networked machines.
- **`--offline --local-data-dir PATH`**: trains from a manually staged local parquet chunk. Intended for Star compute nodes with no outbound network access.
- The preferred Star workflow is one staged sample config at a time. `sample-10BT` is the smoke target, `sample-100BT` is the default serious `1.3b` target.

```bash
# Full sample config via local HuggingFace cache
python train_fineweb_gpt.py -350M --config sample-100BT

# Offline staged chunk (manual downloader workflow)
python train_fineweb_gpt.py -125M \
  --offline \
  --local-data-dir /fs1/proj/educational_web_data/dataset/fineweb-edu/sample-10BT/source \
  --out-dir /fs1/proj/educational_web_data/runs/125m \
  --stop-after-one-epoch

# Offline staged sample chunk for a serious run
python train_fineweb_gpt.py -1.3B \
  --offline \
  --local-data-dir /fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source \
  --out-dir /fs1/proj/educational_web_data/runs/1.3b
```

### Multi-GPU (torchrun)

```bash
./launch_torchrun.sh 350m 8
./launch_torchrun.sh 1.3b 8 /scratch/$USER/fineweb-1.3b
```

### Resume Training

```bash
OUT_DIR=runs/350m
python train_fineweb_gpt.py -350M --out-dir "$OUT_DIR" --resume "$OUT_DIR/fineweb_gpt.ckpt"
```

### Star HPC Offline Workflow

Star compute nodes should not download from HuggingFace directly. Use the login node to stage one sample config at a time.

Smoke-test with `sample-10BT`:

```bash
cd LLM/FineWebEduGPT
python download_fineweb_snapshot.py \
  --config sample-10BT \
  --max-gb 500
```

Serious `1.3b` runs should use `sample-100BT`:

```bash
cd LLM/FineWebEduGPT
python download_fineweb_snapshot.py \
  --config sample-100BT \
  --max-gb 500
```

This populates a staged `source/` directory for the selected sample, for example:

```text
/fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source
```

and advances a downloader state file for that sample so later runs grab the following chunk when needed.

Then submit pretraining:

```bash
LOCAL_DATA_DIRS=/fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source \
sbatch --qos=long2x --export=ALL,OUT_DIR=/fs1/proj/educational_web_data/runs/1.3b,LOCAL_DATA_DIRS="$LOCAL_DATA_DIRS",CONFIGS=sample-100BT \
  -o /fs1/proj/educational_web_data/logs/fineweb-1-3b-%j.out \
  -e /fs1/proj/educational_web_data/logs/fineweb-1-3b-%j.err \
  star_gpu7_fineweb_1_3b.sbatch
```

The Star sbatch files now:
- train on the staged local sample chunk you provided
- save or resume from `<out_dir>/fineweb_gpt.ckpt`
- stop after one full pass over that staged sample chunk
- tell you to run `download_fineweb_snapshot.py` again before the next submit
- use H100-safe `1.3b` defaults of `BATCH_SIZE=1`, `GRAD_ACCUM=128`, and `NO_COMPILE=1`
- auto-clean the disposable local dataset cache unless `KEEP_LOCAL_DATASET_CACHE=1` is set

Smoke-test `1.3b` first with `sample-10BT`:

```bash
LOCAL_DATA_DIRS=/fs1/proj/educational_web_data/dataset/fineweb-edu/sample-10BT/source
sbatch --qos=long2x --export=ALL,OUT_DIR=/fs1/proj/educational_web_data/runs/1.3b-smoke,LOCAL_DATA_DIRS="$LOCAL_DATA_DIRS",CONFIGS=sample-10BT,BATCH_SIZE=1,GRAD_ACCUM=32,NO_COMPILE=1,TRAIN_STEPS=20,EVAL_EVERY=10,EVAL_ITERS=2,CKPT_EVERY=20 \
  -o /fs1/proj/educational_web_data/logs/fineweb-1-3b-smoke-%j.out \
  -e /fs1/proj/educational_web_data/logs/fineweb-1-3b-smoke-%j.err \
  star_gpu7_fineweb_1_3b.sbatch
```

Then launch the full `1.3b` run with `sample-100BT`:

```bash
LOCAL_DATA_DIRS=/fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source
sbatch --qos=long2x --export=ALL,OUT_DIR=/fs1/proj/educational_web_data/runs/1.3b,LOCAL_DATA_DIRS="$LOCAL_DATA_DIRS",CONFIGS=sample-100BT \
  -o /fs1/proj/educational_web_data/logs/fineweb-1-3b-%j.out \
  -e /fs1/proj/educational_web_data/logs/fineweb-1-3b-%j.err \
  star_gpu7_fineweb_1_3b.sbatch
```

The `1.3b` sbatch defaults are now:
- `BATCH_SIZE=1`
- `GRAD_ACCUM=128`
- `NO_COMPILE=1`

That keeps the original `1.3b` global tokens-per-step budget while avoiding the CUDA OOM pattern seen with larger per-GPU batches.

Tokenizer behavior in offline mode:
- if `<out_dir>/tokenizer.model` already exists, it is reused
- otherwise the tokenizer is built from the staged local parquet chunk

Required paths for the `1.3b` preset:

```text
/fs1/proj/educational_web_data/runs/1.3b/tokenizer.model
/fs1/proj/educational_web_data/runs/1.3b/fineweb_gpt.ckpt
/fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source/**/*.parquet
```

## Chat Finetuning

After pretraining completes, run SFT on UltraChat:

```bash
OUT_DIR=runs/350m

# Full finetuning (3 epochs)
python finetune_chat.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt"

# Quick test run (5k samples, ~10 minutes)
python finetune_chat.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --max-samples 5000

# Custom hyperparameters
python finetune_chat.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --epochs 3 --lr 2e-5 --batch-size 4 --grad-accum 8
```

### Custom Dataset

You can also finetune on your own data in JSONL format:

```bash
OUT_DIR=runs/350m
python finetune_chat.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --dataset custom --data-path my_chats.jsonl
```

Each line should be:
```json
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4."}]}
```

Multi-turn conversations are supported -- just add more user/assistant pairs to the list.

### How Masked Loss Works

During finetuning, the loss function only penalizes the model on tokens it should learn to generate (assistant responses). Given this conversation:

```
### User:
What is the capital of France?
### Assistant:
The capital of France is Paris.</s>
```

The loss mask looks like:
```
[0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]
 ^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^
 user turn (masked out)       assistant turn (loss computed here)
```

This prevents the model from wasting capacity trying to predict user inputs and focuses all learning on producing good responses.

## Chat

```bash
OUT_DIR=runs/350m

# With finetuned checkpoint (recommended)
python chat_fineweb_gpt.py --ckpt "$OUT_DIR/fineweb_gpt_chat.ckpt"

# With pretrained checkpoint (raw completion, no chat structure)
python chat_fineweb_gpt.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --raw
```

In-chat commands:
- `/quit` -- exit
- `/clear` -- reset conversation history
- `/raw` -- switch to raw completion mode
- `/chat` -- switch to chat mode

Options:
```bash
--max-tokens 256    # max generation length
--temp 0.7          # sampling temperature
--top-p 0.9         # nucleus sampling threshold
--max-history 5     # conversation turns to keep in context
```

## Evaluation

This repo now includes a lightweight eval stack under `eval/` plus a fixed chat regression set under `eval_data/`.

Quick start:

```bash
OUT_DIR=runs/350m

# LM sanity check
python eval/eval_lm.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --dataset wikitext_valid

# Core multiple-choice benchmarks
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench hellaswag
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench piqa
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench winogrande
python eval/eval_mcq.py --ckpt "$OUT_DIR/fineweb_gpt.ckpt" --bench arc_challenge

# Chat regression suite
python eval/eval_chat.py --ckpt "$OUT_DIR/fineweb_gpt_chat.ckpt" --prompts eval_data/chat_eval_prompts.jsonl

# Convenience wrapper (derives fineweb_gpt.ckpt from the same OUT_DIR)
bash run_eval.sh "$OUT_DIR/fineweb_gpt_chat.ckpt"
```

Contamination scanning:

```bash
python eval/contamination_scan.py \
  --bench hellaswag \
  --bench piqa \
  --fineweb-config sample-100BT \
  --fineweb-sample-docs 5000 \
  --ckpt-label fineweb-scan
```

See [`docs/EVAL.md`](docs/EVAL.md) for the full workflow, interpretation notes, and contamination methodology.

## Artifacts

These live under your chosen `OUT_DIR` (default: `runs/<preset>/`).

| File | Description |
|------|-------------|
| `tokenizer.model` | SentencePiece BPE tokenizer (auto-built from seed data if missing) |
| `fineweb_gpt.ckpt` | Pretrained checkpoint |
| `fineweb_gpt_chat.ckpt` | Best finetuned checkpoint (by val loss) |
| `fineweb_gpt_chat_stepN.ckpt` | Periodic finetuning checkpoints |

## Notes

- Windows skips `torch.compile` to avoid Triton issues.
- The tokenizer is built automatically from staged local parquet if `<out_dir>/tokenizer.model` doesn't exist; otherwise provide a matching tokenizer explicitly.
- The chat format uses regular subword tokens (`### User:`, `### Assistant:`) rather than special tokens, so no vocabulary expansion is needed after pretraining.
- `chat_fineweb_gpt.py` and `finetune_chat.py` auto-resolve `tokenizer.model` from the checkpoint directory unless you override `--tok`.

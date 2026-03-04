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
```

## Pretraining

Default run (single GPU, 350M preset):

```bash
python train_fineweb_gpt.py -350M
```

Pick a crawl snapshot:

```bash
python train_fineweb_gpt.py -350M --config CC-MAIN-2025-26
```

### Size-Based Presets (8x H100 80GB)

| Flag | Parameters | Layers | Heads | Embed | Context | Batch | Grad Accum |
|------|-----------|--------|-------|-------|---------|-------|------------|
| `-125M` | ~125M | 12 | 12 | 768 | 2048 | 64 | 2 |
| `-350M` | ~356M | 24 | 16 | 1024 | 2048 | 32 | 4 |
| `-760M` | ~760M | 24 | 16 | 1536 | 2048 | 16 | 8 |
| `-1.3B` | ~1.3B | 29 | 16 | 1856 | 2048 | 8 | 16 |

```bash
python train_fineweb_gpt.py -125M
python train_fineweb_gpt.py -350M
python train_fineweb_gpt.py -1.3B
```

### Data Loading Modes

- **Default**: downloads the full dataset to local cache, then trains from memory-mapped Arrow files. Fastest throughput.
- **`--stream`**: reads directly from HuggingFace Hub. No disk usage, but slower and network-dependent.
- **`--cache-gb N`**: rolling cache mode. Downloads N GB of data, trains on it, deletes it, downloads the next chunk. Good for limited disk space.

```bash
# Full download (run once, then train)
python train_fineweb_gpt.py -350M

# Streaming (no local storage needed)
python train_fineweb_gpt.py -350M --stream

# Rolling cache (5 GB at a time)
python train_fineweb_gpt.py -350M --cache-gb 5
```

### Multi-GPU (torchrun)

```bash
./launch_torchrun.sh 350m 8
./launch_torchrun.sh 1.3b 8
```

### Resume Training

```bash
python train_fineweb_gpt.py -350M --resume fineweb_gpt.ckpt
```

## Chat Finetuning

After pretraining completes, run SFT on UltraChat:

```bash
# Full finetuning (3 epochs)
python finetune_chat.py --ckpt fineweb_gpt.ckpt --tok tokenizer.model

# Quick test run (5k samples, ~10 minutes)
python finetune_chat.py --ckpt fineweb_gpt.ckpt --max-samples 5000

# Custom hyperparameters
python finetune_chat.py --ckpt fineweb_gpt.ckpt --epochs 3 --lr 2e-5 --batch-size 4 --grad-accum 8
```

### Custom Dataset

You can also finetune on your own data in JSONL format:

```bash
python finetune_chat.py --ckpt fineweb_gpt.ckpt --dataset custom --data-path my_chats.jsonl
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
# With finetuned checkpoint (recommended)
python chat_fineweb_gpt.py --ckpt fineweb_gpt_chat.ckpt --tok tokenizer.model

# With pretrained checkpoint (raw completion, no chat structure)
python chat_fineweb_gpt.py --ckpt fineweb_gpt.ckpt --raw
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

## Artifacts

| File | Description |
|------|-------------|
| `tokenizer.model` | SentencePiece BPE tokenizer (auto-built from seed data if missing) |
| `fineweb_gpt.ckpt` | Pretrained checkpoint |
| `fineweb_gpt_chat.ckpt` | Best finetuned checkpoint (by val loss) |
| `fineweb_gpt_chat_stepN.ckpt` | Periodic finetuning checkpoints |

## Notes

- Windows skips `torch.compile` to avoid Triton issues.
- The tokenizer is built automatically from the first N streamed documents if `tokenizer.model` doesn't exist.
- Full-streaming mode avoids downloading the entire FineWeb-Edu corpus to disk.
- The chat format uses regular subword tokens (`### User:`, `### Assistant:`) rather than special tokens, so no vocabulary expansion is needed after pretraining.

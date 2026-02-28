# FineWebEduGPT

Full-streaming GPT trainer on FineWeb-Edu.

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", "CC-MAIN-2025-26", streaming=True)
```

## What changed

- Uses **streaming dataset reads** (no giant local corpus download).
- Tokenizes on the fly for training batches.
- Supports hardware presets via short flags.
- Includes cluster launcher for `torchrun`.

## Setup

```bash
pip install -r requirements.txt
```

## Train (single GPU)

Default run:

```bash
python train_fineweb_gpt.py
```

Pick crawl snapshot:

```bash
python train_fineweb_gpt.py --config CC-MAIN-2025-26
```

### Presets

- `-5080` → RTX 5080-oriented config
- `-HPC` → generic larger HPC config
- `-10B` → 10B-target preset
- `-A100` → A100 cluster preset
- `-H100` → H100 cluster preset

Examples:

```bash
python train_fineweb_gpt.py -5080
python train_fineweb_gpt.py -10B
python train_fineweb_gpt.py -A100
python train_fineweb_gpt.py -H100
```

Equivalent long form:

```bash
python train_fineweb_gpt.py --preset h100
```

## Train (multi-GPU with torchrun)

```bash
./launch_torchrun.sh h100 8
./launch_torchrun.sh a100 4

torchrun --standalone --nproc_per_node=8 train_fineweb_gpt.py --cache-gb 5 --eval-every 5 --eval-iters 12 --batch-size 64 --context 1024 --grad-accum 4 --num-workers 8 --queue-size 256 --resume fineweb_gpt.ckpt
```

Arguments are:
1. preset (`h100`, `a100`, `10b`, `hpc`, `5080`)
2. processes per node (GPU count)

## Chat

```bash
python chat_fineweb_gpt.py
```

## Artifacts

- `tokenizer.model` (built once if missing)
- `fineweb_gpt.ckpt` (training checkpoint)
- Optional `tokenizer_seed.txt` (small seed text used only when tokenizer must be created)

## Notes

- Windows skips `torch.compile` to avoid Triton issues.
- Full-streaming means you are not downloading the full FineWeb-Edu corpus to disk.

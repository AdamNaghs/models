# FineWebEduGPT

This folder trains a GPT-style model on FineWeb-Edu using:

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", "CC-MAIN-2013-48")
```

Default config is **CC-MAIN-2013-48** (newer crawl slice than 2013-20), with 2013-20 available via flag.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train_fineweb_gpt.py
```

Optional (switch slice):

```bash
python train_fineweb_gpt.py --config CC-MAIN-2013-20
```

Useful knobs:

```bash
python train_fineweb_gpt.py --samples 200000 --train-steps 30000 --batch-size 24
```

## Chat

```bash
python chat_fineweb_gpt.py
```

## Notes

- First run downloads and builds:
  - `fineweb_text.txt`
  - `tokenizer.model`
  - `train_tokens.npy`
- Training checkpoint: `fineweb_gpt.ckpt`
- On Windows, `torch.compile` is skipped to avoid Triton issues.

# GPT Stinker (BPE + ~80M class model scaffold)

This is the next-step stack beyond `stinker.py`:
- BPE tokenizer (SentencePiece)
- Token preprocessing to `.npy`
- GPT-style training (`train_gpt.py`)
- Multi-turn chat runner (`chat_gpt.py`)

## 1) Install deps

```bash
pip install -r requirements.txt
```

## 2) Build tokenizer

```bash
python train_tokenizer.py
```

Outputs:
- `tokenizer.model`
- `tokenizer.vocab`

## 3) Encode dataset to token ids

```bash
python preprocess_tokens.py
```

Output:
- `train_tokens.npy`

## 4) Train model

```bash
python train_gpt.py
```

Output:
- `gpt_stinker.ckpt`

## 5) Chat with model

```bash
python chat_gpt.py
```

## Notes

- Default config in `train_gpt.py` is a realistic starter for a single modern GPU.
- On Windows, `torch.compile` is skipped where it tends to fail with Triton issues.
- Adjust `BATCH_SIZE` first if you hit out-of-memory.

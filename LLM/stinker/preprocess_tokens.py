from pathlib import Path
import numpy as np
import sentencepiece as spm

DATA_PATH = Path("data.txt")
TOKENIZER_PATH = Path("tokenizer.model")
TOKENS_PATH = Path("train_tokens.npy")

if not DATA_PATH.exists():
    raise FileNotFoundError("data.txt not found")
if not TOKENIZER_PATH.exists():
    raise FileNotFoundError("tokenizer.model not found; run train_tokenizer.py first")

text = DATA_PATH.read_text(encoding="utf-8")
sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
ids = np.array(sp.encode(text, out_type=int), dtype=np.uint16)
np.save(TOKENS_PATH, ids)

print(f"Saved {TOKENS_PATH} with {len(ids):,} tokens")

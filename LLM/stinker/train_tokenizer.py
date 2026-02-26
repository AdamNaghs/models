from pathlib import Path
import sentencepiece as spm

DATA_PATH = Path("data.txt")
MODEL_PREFIX = "tokenizer"
VOCAB_SIZE = 8192

if not DATA_PATH.exists():
    raise FileNotFoundError("data.txt not found in current directory")

spm.SentencePieceTrainer.train(
    input=str(DATA_PATH),
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=1.0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    unk_id=0,
    user_defined_symbols=["<|user|>", "<|assistant|>", "<|end|>"],
)

print("Wrote tokenizer.model and tokenizer.vocab")

from pathlib import Path
import multiprocessing as mp
import numpy as np
import sentencepiece as spm

DATA_PATH = Path("data.txt")
TOKENIZER_PATH = Path("tokenizer.model")
TOKENS_PATH = Path("train_tokens.npy")

# Tune these on your machine
CHUNK_CHARS = 2_000_000
WORKERS = max(1, mp.cpu_count() - 1)

_sp = None


def _init_worker(tokenizer_path: str):
    global _sp
    _sp = spm.SentencePieceProcessor(model_file=tokenizer_path)


def _encode_chunk(text_chunk: str):
    ids = _sp.encode(text_chunk, out_type=int)
    return np.asarray(ids, dtype=np.uint16)


def chunked_text(path: Path, chunk_chars: int):
    buf = []
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            n += len(line)
            if n >= chunk_chars:
                yield "".join(buf)
                buf, n = [], 0
    if buf:
        yield "".join(buf)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("data.txt not found")
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("tokenizer.model not found; run train_tokenizer.py first")

    chunks = list(chunked_text(DATA_PATH, CHUNK_CHARS))
    total_chunks = len(chunks)
    if total_chunks == 0:
        np.save(TOKENS_PATH, np.asarray([], dtype=np.uint16))
        print(f"Saved {TOKENS_PATH} with 0 tokens")
        return

    print(f"Encoding {total_chunks} chunks with {WORKERS} workers...")

    encoded_parts = []
    with mp.Pool(processes=WORKERS, initializer=_init_worker, initargs=(str(TOKENIZER_PATH),)) as pool:
        for i, arr in enumerate(pool.imap(_encode_chunk, chunks, chunksize=1), 1):
            encoded_parts.append(arr)
            if i % 10 == 0 or i == total_chunks:
                print(f"chunk {i}/{total_chunks}")

    ids = np.concatenate(encoded_parts, axis=0) if encoded_parts else np.asarray([], dtype=np.uint16)
    np.save(TOKENS_PATH, ids)
    print(f"Saved {TOKENS_PATH} with {len(ids):,} tokens")


if __name__ == "__main__":
    mp.freeze_support()  # Windows-safe multiprocessing startup
    main()

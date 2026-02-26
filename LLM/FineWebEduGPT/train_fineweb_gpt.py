import argparse
import itertools
import math
import platform
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train a GPT on FineWeb-Edu")
    p.add_argument("--config", default="CC-MAIN-2013-48", choices=["CC-MAIN-2013-48", "CC-MAIN-2013-20"])
    p.add_argument("--samples", type=int, default=120000, help="Number of FineWeb-Edu samples to pull")
    p.add_argument("--train-steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=10)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--grad-accum", type=int, default=4, help="Microbatches per optimizer step")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--eval-iters", type=int, default=10)
    return p.parse_args()


def ensure_text(args, text_path: Path):
    if text_path.exists() and text_path.stat().st_size > 0:
        print(f"Using cached text: {text_path}")
        return

    print(f"Downloading FineWeb-Edu {args.config} (samples={args.samples:,})...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", args.config, split="train", streaming=True)

    kept = 0
    with text_path.open("w", encoding="utf-8") as f:
        for ex in itertools.islice(ds, args.samples):
            txt = (ex.get("text") or "").strip()
            if not txt:
                continue
            f.write(txt + "\n")
            kept += 1

    print(f"Wrote {kept:,} documents to {text_path}")


def ensure_tokenizer(text_path: Path, tok_model: Path, vocab_size: int):
    if tok_model.exists():
        print("Using cached tokenizer.model")
        return

    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix="tokenizer",
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_id=0,
    )


def ensure_tokens(text_path: Path, tok_model: Path, tok_npy: Path):
    if tok_npy.exists() and tok_npy.stat().st_size > 0:
        print("Using cached train_tokens.npy")
        return

    print("Encoding text to token ids...")
    sp = spm.SentencePieceProcessor(model_file=str(tok_model))
    text = text_path.read_text(encoding="utf-8")
    ids = np.asarray(sp.encode(text, out_type=int), dtype=np.uint16)
    np.save(tok_npy, ids)
    print(f"Saved {tok_npy} with {len(ids):,} tokens")


class RandomTokenDataset(Dataset):
    def __init__(self, arr: np.ndarray, context: int):
        self.arr = arr
        self.context = context
        self.max_i = len(arr) - context - 1
        if self.max_i <= 0:
            raise ValueError("Not enough tokens for selected context length")

    def __len__(self):
        return 10_000_000

    def __getitem__(self, _):
        i = np.random.randint(0, self.max_i)
        x = torch.from_numpy(self.arr[i : i + self.context].astype(np.int64, copy=False))
        y = torch.from_numpy(self.arr[i + 1 : i + 1 + self.context].astype(np.int64, copy=False))
        return x, y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        t = x.size(1)
        m = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=m, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, context, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(context, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    text_path = Path("fineweb_text.txt")
    tok_model = Path("tokenizer.model")
    tok_npy = Path("train_tokens.npy")
    ckpt_path = Path("fineweb_gpt.ckpt")

    ensure_text(args, text_path)
    ensure_tokenizer(text_path, tok_model, args.vocab_size)
    ensure_tokens(text_path, tok_model, tok_npy)

    data = np.load(tok_npy, mmap_mode="r")
    n = len(data)
    n_train = int(n * 0.98)
    train_np, val_np = data[:n_train], data[n_train:]

    workers = max(0, args.num_workers)
    train_loader = DataLoader(
        RandomTokenDataset(train_np, args.context),
        batch_size=args.batch_size,
        num_workers=workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(workers > 0),
        prefetch_factor=args.prefetch_factor if workers > 0 else None,
    )
    val_loader = DataLoader(
        RandomTokenDataset(val_np, args.context),
        batch_size=args.batch_size,
        num_workers=max(0, workers // 2),
        pin_memory=(device == "cuda"),
        persistent_workers=(workers // 2 > 0),
        prefetch_factor=args.prefetch_factor if workers // 2 > 0 else None,
    )
    train_it = iter(train_loader)
    val_it = iter(val_loader)

    def next_batch(which="train"):
        nonlocal train_it, val_it
        it = train_it if which == "train" else val_it
        loader = train_loader if which == "train" else val_loader
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        if which == "train":
            train_it = it
        else:
            val_it = it

        if device == "cuda":
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    vocab = spm.SentencePieceProcessor(model_file=str(tok_model)).vocab_size()
    model = GPT(vocab, args.context, args.n_embd, args.n_head, args.n_layer, args.dropout).to(device)

    if device == "cuda" and platform.system() != "Windows":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    use_fused = device == "cuda"
    try:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=use_fused,
        )
    except TypeError:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    @torch.no_grad()
    def eval_loss(iters=10):
        model.eval()
        out = {}
        for split in ("train", "val"):
            vals = []
            for _ in range(iters):
                xb, yb = next_batch(split)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    _, loss = model(xb, yb)
                vals.append(loss.item())
            out[split] = sum(vals) / len(vals)
        model.train()
        return out

    params = sum(p.numel() for p in model.parameters())
    eff_tokens = args.batch_size * args.context * args.grad_accum
    print(
        f"device={device} | vocab={vocab} | params={params:,} | config={args.config} | "
        f"workers={workers} | grad_accum={args.grad_accum} | eff_tokens/step={eff_tokens:,}"
    )

    for step in range(args.train_steps + 1):
        if step % args.eval_every == 0:
            losses = eval_loss(args.eval_iters)
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | ppl {math.exp(losses['val']):.2f}"
            )

        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            xb, yb = next_batch("train")
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                _, loss = model(xb, yb)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()

        scaler.step(opt)
        scaler.update()

        if step > 0 and step % args.ckpt_every == 0:
            sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save({"state_dict": sd, "args": vars(args), "vocab": vocab}, ckpt_path)
            print(f"checkpoint -> {ckpt_path}")

    sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({"state_dict": sd, "args": vars(args), "vocab": vocab}, ckpt_path)
    print(f"saved -> {ckpt_path}")


if __name__ == "__main__":
    main()

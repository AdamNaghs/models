import argparse
import itertools
import math
import platform
import queue
import threading
from collections import deque

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Full-streaming GPT trainer on FineWeb-Edu")
    p.add_argument(
        "--config",
        default="CC-MAIN-2025-26",
        choices=[
            "CC-MAIN-2025-05",
            "CC-MAIN-2025-08",
            "CC-MAIN-2025-13",
            "CC-MAIN-2025-18",
            "CC-MAIN-2025-21",
            "CC-MAIN-2025-26",
        ],
    )
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
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--tokenizer-model", default="tokenizer.model")
    p.add_argument("--queue-size", type=int, default=64)
    p.add_argument("--seed-docs", type=int, default=5000, help="Docs used once to build tokenizer if missing")
    return p.parse_args()


def ensure_tokenizer(args):
    tok_model = args.tokenizer_model
    if platform.system() == "Windows":
        prefix = tok_model.replace(".model", "")
    else:
        prefix = tok_model[:-6] if tok_model.endswith(".model") else tok_model

    try:
        sp = spm.SentencePieceProcessor(model_file=tok_model)
        print(f"Using existing tokenizer: {tok_model}")
        return sp
    except Exception:
        pass

    print(f"Tokenizer missing. Building from first {args.seed_docs:,} streamed docs...")
    seed_path = "tokenizer_seed.txt"
    ds = load_dataset("HuggingFaceFW/fineweb-edu", args.config, split="train", streaming=True)
    with open(seed_path, "w", encoding="utf-8") as f:
        for ex in itertools.islice(ds, args.seed_docs):
            t = (ex.get("text") or "").strip()
            if t:
                f.write(t + "\n")

    spm.SentencePieceTrainer.train(
        input=seed_path,
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_id=0,
    )
    return spm.SentencePieceProcessor(model_file=tok_model)


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


class StreamingBatcher:
    def __init__(self, sp, config, context, batch_size, queue_size=64):
        self.sp = sp
        self.context = context
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=queue_size)
        self.stop = threading.Event()
        self.config = config
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        ds = load_dataset("HuggingFaceFW/fineweb-edu", self.config, split="train", streaming=True)
        token_buf = deque()
        for ex in ds:
            if self.stop.is_set():
                break
            txt = (ex.get("text") or "").strip()
            if not txt:
                continue
            ids = self.sp.encode(txt, out_type=int)
            token_buf.extend(ids)

            needed = self.batch_size * (self.context + 1)
            while len(token_buf) >= needed and not self.stop.is_set():
                block = [token_buf.popleft() for _ in range(needed)]
                t = torch.tensor(block, dtype=torch.long).view(self.batch_size, self.context + 1)
                x = t[:, :-1].contiguous()
                y = t[:, 1:].contiguous()
                try:
                    self.q.put((x, y), timeout=1.0)
                except queue.Full:
                    if self.stop.is_set():
                        break
                    continue

    def next(self, device):
        x, y = self.q.get(timeout=120)
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    def close(self):
        self.stop.set()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    sp = ensure_tokenizer(args)
    vocab = sp.vocab_size()

    train_stream = StreamingBatcher(sp, args.config, args.context, args.batch_size, queue_size=args.queue_size)
    val_stream = StreamingBatcher(sp, args.config, args.context, args.batch_size, queue_size=max(16, args.queue_size // 2))

    model = GPT(vocab, args.context, args.n_embd, args.n_head, args.n_layer, args.dropout).to(device)
    if device == "cuda" and platform.system() != "Windows":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    try:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=(device == "cuda"),
        )
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    @torch.no_grad()
    def eval_loss(iters):
        model.eval()
        vals = []
        for _ in range(iters):
            xb, yb = val_stream.next(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                _, loss = model(xb, yb)
            vals.append(loss.item())
        model.train()
        return sum(vals) / len(vals)

    params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device} | vocab={vocab} | params={params:,} | config={args.config} | "
        f"grad_accum={args.grad_accum} | full-streaming=yes"
    )

    ckpt_path = "fineweb_gpt.ckpt"

    try:
        for step in range(args.train_steps + 1):
            if step % args.eval_every == 0:
                v = eval_loss(args.eval_iters)
                print(f"step {step:5d} | val {v:.4f} | ppl {math.exp(v):.2f}")

            opt.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                xb, yb = train_stream.next(device)
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
    finally:
        train_stream.close()
        val_stream.close()


if __name__ == "__main__":
    main()

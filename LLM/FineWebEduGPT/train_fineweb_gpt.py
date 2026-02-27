import argparse
import itertools
import math
import platform
import queue
import sys
import threading
import time
from collections import deque

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset


PRESETS = {
    "5080": {
        "train_steps": 150000,
        "batch_size": 2,
        "context": 1024,
        "n_layer": 20,
        "n_head": 16,
        "n_embd": 1024,
        "grad_accum": 32,
        "vocab_size": 32000,
        "eval_every": 500,
        "eval_iters": 12,
        "ckpt_every": 2000,
        "seed_docs": 100000,
    },
    "hpc": {
        "train_steps": 200000,
        "batch_size": 16,
        "context": 2048,
        "n_layer": 32,
        "n_head": 32,
        "n_embd": 4096,
        "grad_accum": 8,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 200000,
    },
    "10b": {
        "train_steps": 250000,
        "batch_size": 8,
        "context": 2048,
        "n_layer": 40,
        "n_head": 40,
        "n_embd": 3840,
        "grad_accum": 16,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 250000,
    },
    "a100": {
        "train_steps": 250000,
        "batch_size": 24,
        "context": 2048,
        "n_layer": 40,
        "n_head": 40,
        "n_embd": 3840,
        "grad_accum": 6,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 250000,
    },
    "h100": {
        "train_steps": 300000,
        "batch_size": 32,
        "context": 4096,
        "n_layer": 48,
        "n_head": 48,
        "n_embd": 4608,
        "grad_accum": 4,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 300000,
    },
}


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
    p.add_argument("--train-steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=10)
    p.add_argument("--n-embd", type=int, default=640)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=12)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--tokenizer-model", default="tokenizer.model")
    p.add_argument("--queue-size", type=int, default=64)
    p.add_argument("--seed-docs", type=int, default=50000, help="Docs used once to build tokenizer if missing")
    p.add_argument("--preset", choices=["5080", "hpc", "10b", "a100", "h100"], help="Apply a training preset")

    g = p.add_mutually_exclusive_group()
    g.add_argument("-5080", dest="preset_5080", action="store_true", help="Use RTX 5080-oriented preset")
    g.add_argument("-HPC", dest="preset_hpc", action="store_true", help="Use large HPC preset")
    g.add_argument("-10B", dest="preset_10b", action="store_true", help="Use 10B-target preset")
    g.add_argument("-A100", dest="preset_a100", action="store_true", help="Use A100 cluster preset")
    g.add_argument("-H100", dest="preset_h100", action="store_true", help="Use H100 cluster preset")

    args = p.parse_args()

    if args.preset_5080:
        args.preset = "5080"
    elif args.preset_hpc:
        args.preset = "hpc"
    elif args.preset_10b:
        args.preset = "10b"
    elif args.preset_a100:
        args.preset = "a100"
    elif args.preset_h100:
        args.preset = "h100"

    # Precedence: explicit CLI flags > preset > parser defaults
    explicit_flags = set(a.split("=")[0] for a in sys.argv[1:] if a.startswith("-"))
    key_to_flags = {
        "train_steps": {"--train-steps"},
        "batch_size": {"--batch-size"},
        "context": {"--context"},
        "n_layer": {"--n-layer"},
        "n_head": {"--n-head"},
        "n_embd": {"--n-embd"},
        "dropout": {"--dropout"},
        "lr": {"--lr"},
        "weight_decay": {"--weight-decay"},
        "eval_every": {"--eval-every"},
        "eval_iters": {"--eval-iters"},
        "ckpt_every": {"--ckpt-every"},
        "grad_accum": {"--grad-accum"},
        "vocab_size": {"--vocab-size"},
        "tokenizer_model": {"--tokenizer-model"},
        "queue_size": {"--queue-size"},
        "seed_docs": {"--seed-docs"},
    }

    if args.preset:
        for k, v in PRESETS[args.preset].items():
            if not (key_to_flags.get(k, set()) & explicit_flags):
                setattr(args, k, v)

    return args


def estimate_params(vocab, context, n_embd, n_layer):
    # rough GPT estimate (embeddings + blocks + lm head)
    return (
        vocab * n_embd  # token embedding
        + context * n_embd  # position embedding
        + n_layer * (12 * n_embd * n_embd + 13 * n_embd)  # attention + MLP + norms
        + n_embd * vocab  # lm head
    )


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
    est = estimate_params(vocab, args.context, args.n_embd, args.n_layer)
    print(
        f"device={device} | preset={args.preset or 'custom'} | vocab={vocab} | params={params:,} "
        f"(est {est:,}) | config={args.config} | grad_accum={args.grad_accum} | full-streaming=yes"
    )

    ckpt_path = "fineweb_gpt.ckpt"
    start_time = time.perf_counter()
    last_step_time = start_time

    try:
        for step in range(args.train_steps + 1):
            now = time.perf_counter()
            step_gap = now - last_step_time if step > 0 else 0.0
            elapsed = now - start_time
            avg_step = elapsed / max(step, 1)
            eta = max(args.train_steps - step, 0) * avg_step

            if step % args.eval_every == 0:
                v = eval_loss(args.eval_iters)
                print(
                    f"step {step:5d} | val {v:.4f} | ppl {math.exp(v):.2f} | "
                    f"dt {step_gap:.2f}s | elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m"
                )

            opt.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                xb, yb = train_stream.next(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    _, loss = model(xb, yb)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            last_step_time = time.perf_counter()

            if step > 0 and step % args.ckpt_every == 0:
                sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
                torch.save({"state_dict": sd, "args": vars(args), "vocab": vocab}, ckpt_path)
                print(f"checkpoint -> {ckpt_path} | elapsed {(time.perf_counter()-start_time)/60:.1f}m")

        sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        torch.save({"state_dict": sd, "args": vars(args), "vocab": vocab}, ckpt_path)
        print(f"saved -> {ckpt_path}")
    finally:
        train_stream.close()
        val_stream.close()


if __name__ == "__main__":
    main()

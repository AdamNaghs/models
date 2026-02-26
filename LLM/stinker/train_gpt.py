import math
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- config (RTX 5080-friendly starter) --------
BATCH_SIZE = 24
CONTEXT = 512
N_LAYER = 10
N_HEAD = 8
N_EMBD = 512
DROPOUT = 0.1
STEPS = 12000
LR = 3e-4
WD = 0.1
EVAL_EVERY = 200
CKPT_EVERY = 1000

TOKENS_PATH = Path("train_tokens.npy")
TOKENIZER_PATH = Path("tokenizer.model")
CKPT_PATH = Path("gpt_stinker.ckpt")

# ----------------------------------------------------

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision("high")

if not TOKENS_PATH.exists():
    raise FileNotFoundError("train_tokens.npy not found; run preprocess_tokens.py")
if not TOKENIZER_PATH.exists():
    raise FileNotFoundError("tokenizer.model not found; run train_tokenizer.py")

sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
V = sp.vocab_size()

data = np.load(TOKENS_PATH, mmap_mode="r")
if data.dtype != np.uint16:
    data = data.astype(np.uint16)

n = len(data)
n_train = int(n * 0.98)
train_np = data[:n_train]
val_np = data[n_train:]


def get_batch(split="train"):
    src = train_np if split == "train" else val_np
    max_i = len(src) - CONTEXT - 1
    ix = torch.randint(0, max_i, (BATCH_SIZE,), device="cpu")

    x = torch.stack([
        torch.from_numpy(src[i : i + CONTEXT].astype(np.int64, copy=False)) for i in ix.tolist()
    ])
    y = torch.stack([
        torch.from_numpy(src[i + 1 : i + 1 + CONTEXT].astype(np.int64, copy=False)) for i in ix.tolist()
    ])

    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, dropout=DROPOUT, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)[0]
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, N_EMBD)
        self.pos = nn.Embedding(CONTEXT, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, V, bias=False)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = F.cross_entropy(logits.view(-1, V), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temp=0.8, top_p=0.9):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -CONTEXT:])
            logits = logits[:, -1, :] / max(temp, 1e-4)
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            csum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = csum > top_p
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_i = torch.multinomial(sorted_probs, 1)
            next_tok = torch.gather(sorted_idx, -1, next_i)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


@torch.no_grad()
def eval_loss(model, iters=20):
    model.eval()
    losses = {}
    for split in ("train", "val"):
        vals = []
        for _ in range(iters):
            x, y = get_batch(split)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                _, loss = model(x, y)
            vals.append(loss.item())
        losses[split] = sum(vals) / len(vals)
    model.train()
    return losses


model = GPT().to(device)
if device == "cuda" and torch.cuda.is_available() and __import__("platform").system() != "Windows":
    try:
        model = torch.compile(model)
        print("compiled model")
    except Exception:
        print("compile skipped")

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WD)
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

params = sum(p.numel() for p in model.parameters())
print(f"device={device} | vocab={V} | params={params:,}")

for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0:
        losses = eval_loss(model, iters=10)
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | ppl {math.exp(losses['val']):.2f}")

    xb, yb = get_batch("train")
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        _, loss = model(xb, yb)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    if step > 0 and step % CKPT_EVERY == 0:
        state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        torch.save(
            {
                "state_dict": state_dict,
                "config": {
                    "context": CONTEXT,
                    "n_layer": N_LAYER,
                    "n_head": N_HEAD,
                    "n_embd": N_EMBD,
                    "dropout": DROPOUT,
                    "vocab": V,
                },
            },
            CKPT_PATH,
        )
        print(f"checkpoint -> {CKPT_PATH}")

# final save
state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
torch.save(
    {
        "state_dict": state_dict,
        "config": {
            "context": CONTEXT,
            "n_layer": N_LAYER,
            "n_head": N_HEAD,
            "n_embd": N_EMBD,
            "dropout": DROPOUT,
            "vocab": V,
        },
    },
    CKPT_PATH,
)
print(f"saved -> {CKPT_PATH}")

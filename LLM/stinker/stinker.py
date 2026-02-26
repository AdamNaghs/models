import json
import os
import platform
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numpy as np
except Exception:
    np = None


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision("high")

USER_TAG = "<|user|>"
ASSIST_TAG = "<|assistant|>"
END_TAG = "<|end|>"

DATA_TXT = Path("data.txt")
CACHE_META = Path("data_cache_meta.json")
CACHE_TOKENS = Path("data_tokens_u16.npy")
CACHE_TOKENS_PT = Path("data_tokens_u16.pt")

B, T = 32, 256
N, H, L = 256, 8, 6


def build_chat_text(raw_text: str) -> str:
    if USER_TAG not in raw_text or ASSIST_TAG not in raw_text:
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not lines:
            lines = ["hello", "small llm demo"]
        convo = []
        for i in range(0, len(lines), 2):
            u = lines[i]
            a = lines[i + 1] if i + 1 < len(lines) else lines[i]
            convo.append(f"{USER_TAG} {u}\n{ASSIST_TAG} {a} {END_TAG}\n")
        text = "".join(convo)
    else:
        text = raw_text

    for tok in (USER_TAG, ASSIST_TAG, END_TAG):
        if tok not in text:
            text += f"\n{tok}"
    return text


def data_signature(path: Path):
    if path.exists():
        st = path.stat()
        return {"path": str(path), "size": st.st_size, "mtime": st.st_mtime_ns}
    return {"path": str(path), "size": -1, "mtime": -1}


def load_or_build_tokens():
    sig = data_signature(DATA_TXT)

    if CACHE_META.exists() and (CACHE_TOKENS.exists() or CACHE_TOKENS_PT.exists()):
        try:
            meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
            if meta.get("data_sig") == sig:
                stoi = {k: int(v) for k, v in meta["stoi"].items()}
                itos = {int(k): v for k, v in meta["itos"].items()}
                if CACHE_TOKENS.exists() and np is not None:
                    tokens_u16 = np.load(CACHE_TOKENS, mmap_mode="r")
                    tokens_u16 = torch.from_numpy(tokens_u16.copy())
                    print("Loaded token cache via numpy memmap")
                else:
                    tokens_u16 = torch.load(CACHE_TOKENS_PT, map_location="cpu")
                    print("Loaded token cache via torch")
                return stoi, itos, tokens_u16
        except Exception:
            pass

    try:
        raw_text = DATA_TXT.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raw_text = "small llm demo in under 100 lines. " * 200

    text = build_chat_text(raw_text)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    tokens_u16 = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.uint16, device="cpu")

    meta = {
        "data_sig": sig,
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "dtype": "uint16",
    }
    CACHE_META.write_text(json.dumps(meta), encoding="utf-8")
    torch.save(tokens_u16, CACHE_TOKENS_PT)
    if np is not None:
        np.save(CACHE_TOKENS, tokens_u16.numpy())
    print("Built and cached tokenized data")

    return stoi, itos, tokens_u16


stoi, itos, tokens_u16 = load_or_build_tokens()
V = len(stoi)
enc = lambda s: torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)
dec = lambda t: "".join(itos[i] for i in t.tolist())

# Keep base tokens compact on CPU; cast to long only for sampled batches.
data_u16 = tokens_u16.contiguous()

# One-time train/val split indices.
n_total = len(data_u16)
n_train = int(n_total * 0.9)
train_u16 = data_u16[:n_train]
val_u16 = data_u16[n_train:]


def batch():
    src_u16 = train_u16 if torch.rand(()) < 0.9 else val_u16
    i = torch.randint(len(src_u16) - T - 1, (B,), device="cpu")

    x_cpu = torch.stack([src_u16[j : j + T] for j in i]).to(dtype=torch.long).pin_memory()
    y_cpu = torch.stack([src_u16[j + 1 : j + T + 1] for j in i]).to(dtype=torch.long).pin_memory()

    if device == "cuda":
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
    else:
        x, y = x_cpu, y_cpu
    return x, y


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N), nn.LayerNorm(N)
        self.attn = nn.MultiheadAttention(N, H, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(N, 4 * N),
            nn.GELU(),
            nn.Linear(4 * N, N),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        t = x.size(1)
        m = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=m, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, N)
        self.pos = nn.Embedding(T, N)
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])
        self.ln = nn.LayerNorm(N)
        self.head = nn.Linear(N, V)

    def forward(self, x, y=None):
        _, t = x.shape
        x = self.tok(x) + self.pos(torch.arange(t, device=x.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1)) if y is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, x, n=200, temp=0.8):
        for _ in range(n):
            logits, _ = self(x[:, -T:])
            probs = F.softmax(logits[:, -1] / max(temp, 1e-4), dim=-1)
            x = torch.cat([x, torch.multinomial(probs, 1)], dim=1)
        return x


m = TinyLM().to(device)
if device == "cuda" and torch.cuda.is_available() and platform.system() != "Windows":
    try:
        print("Compiling model")
        m = torch.compile(m)
    except Exception:
        print("Failed to compile model with cuda.")
else:
    print("Skipping compile (cpu or windows)")

opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

total = sum(p.numel() for p in m.parameters())
print(
    "device:",
    device,
    "| cuda:",
    torch.cuda.is_available(),
    "|",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "| params:",
    f"{total:,}",
)

for step in range(4500):
    x, y = batch()
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        _, loss = m(x, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    if step % 200 == 0:
        print(f"step {step:4d} | loss {loss.item():.3f}")

start = enc(f"{USER_TAG} hello\n{ASSIST_TAG} ").unsqueeze(0).to(device)
out = m.generate(start, n=220, temp=0.8)
print("\n--- sample ---\n" + dec(out[0].cpu()))

ckpt = {
    "state_dict": m.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "config": {
        "B": B,
        "T": T,
        "N": N,
        "H": H,
        "L": L,
        "V": V,
        "user_tag": USER_TAG,
        "assistant_tag": ASSIST_TAG,
        "end_tag": END_TAG,
    },
}
torch.save(ckpt, "stinker.ckpt")
print("saved -> stinker.ckpt")

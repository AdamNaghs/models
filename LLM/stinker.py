import math, torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    with open("repos/models/LLM/data.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    text = "small llm demo in under 100 lines. " * 200
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
enc = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
dec = lambda t: "".join(itos[i] for i in t.tolist())
data = enc(text).to(device)

B, T = 32, 64
V = len(chars)
N, H, L = 128, 4, 3


def batch(split=0.9):
    n = int(len(data) * split)
    src = data[:n] if torch.rand(()) < 0.9 else data[n:]
    i = torch.randint(len(src) - T - 1, (B,), device=device)
    x = torch.stack([src[j : j + T] for j in i])
    y = torch.stack([src[j + 1 : j + T + 1] for j in i])
    return x, y


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N), nn.LayerNorm(N)
        self.attn = nn.MultiheadAttention(N, H, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(N, 4 * N), nn.GELU(), nn.Linear(4 * N, N), nn.Dropout(0.1))

    def forward(self, x):
        m = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
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
        b, t = x.shape
        x = self.tok(x) + self.pos(torch.arange(t, device=x.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1)) if y is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, x, n=200, temp=1.0):
        for _ in range(n):
            logits, _ = self(x[:, -T:])
            probs = F.softmax(logits[:, -1] / temp, dim=-1)
            x = torch.cat([x, torch.multinomial(probs, 1)], dim=1)
        return x


m = TinyLM().to(device)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)

for step in range(1200):
    x, y = batch()
    _, loss = m(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"step {step:4d} | loss {loss.item():.3f}")

start = torch.tensor([[stoi["s"]]], device=device)
out = m.generate(start, n=300)
print("\n--- sample ---\n" + dec(out[0].cpu()))

# save
ckpt = {
"state_dict": m.state_dict(),
"stoi": stoi,
"itos": itos,
"config": {"B": B, "T": T, "N": N, "H": H, "L": L, "V": V},
}
torch.save(ckpt, "repos/models/LLM/stinker.ckpt")
print("saved -> repos/models/LLM/stinker.ckpt")

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("repos/models/LLM/stinker.ckpt", map_location=device)

stoi, itos = ckpt["stoi"], ckpt["itos"]
T = ckpt["config"]["T"]
N, H, L, V = ckpt["config"]["N"], ckpt["config"]["H"], ckpt["config"]["L"], ckpt["config"]["V"]

enc = lambda s: torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long, device=device)
dec = lambda t: "".join(itos[i] for i in t.tolist())

class Block(nn.Module):
def __init__(self):
super().__init__()
self.ln1, self.ln2 = nn.LayerNorm(N), nn.LayerNorm(N)
self.attn = nn.MultiheadAttention(N, H, dropout=0.0, batch_first=True)
self.ff = nn.Sequential(nn.Linear(N, 4*N), nn.GELU(), nn.Linear(4*N, N))
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
def forward(self, x):
b, t = x.shape
x = self.tok(x) + self.pos(torch.arange(t, device=x.device))
return self.head(self.ln(self.blocks(x)))
@torch.no_grad()
def generate(self, x, n=120, temp=0.8):
for _ in range(n):
logits = self(x[:, -T:])
probs = F.softmax(logits[:, -1] / temp, dim=-1)
x = torch.cat([x, torch.multinomial(probs, 1)], dim=1)
return x

m = TinyLM().to(device)
m.load_state_dict(ckpt["state_dict"])
m.eval()

print("Stinker chat. Ctrl+C to quit.")
while True:
q = input("\nYou: ")
prompt = f"\nUser: {q}\nStinker:"
x = enc(prompt).unsqueeze(0)
if x.numel() == 0:
print("Stinker: (I don't know these characters yet)")
continue
y = m.generate(x, n=180, temp=0.7)
out = dec(y[0])[len(prompt):]
print("Stinker:", out.split("\n")[0].strip())

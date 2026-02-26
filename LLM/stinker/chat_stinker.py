import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("stinker.ckpt", map_location=device)

stoi, itos = ckpt["stoi"], ckpt["itos"]
cfg = ckpt["config"]
T, N, H, L, V = cfg["T"], cfg["N"], cfg["H"], cfg["L"], cfg["V"]
USER_TAG = cfg.get("user_tag", "<|user|>")
ASSIST_TAG = cfg.get("assistant_tag", "<|assistant|>")
END_TAG = cfg.get("end_tag", "<|end|>")

enc = lambda s: torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long, device=device)
dec = lambda t: "".join(itos[i] for i in t.tolist())


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N), nn.LayerNorm(N)
        self.attn = nn.MultiheadAttention(N, H, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(N, 4 * N), nn.GELU(), nn.Linear(4 * N, N))

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
        _, t = x.shape
        x = self.tok(x) + self.pos(torch.arange(t, device=x.device))
        return self.head(self.ln(self.blocks(x)))

    @torch.no_grad()
    def generate(
        self,
        x,
        n=160,
        temp=0.75,
        top_p=0.9,
        repetition_penalty=1.1,
    ):
        for _ in range(n):
            logits = self(x[:, -T:])[:, -1, :]

            # repetition penalty on recent context
            for token_id in torch.unique(x[:, -min(T, x.size(1)) :]):
                logits[:, token_id] /= repetition_penalty

            probs = F.softmax(logits / max(temp, 1e-4), dim=-1)

            # nucleus (top-p) sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            csum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = csum > top_p
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_idx = torch.multinomial(sorted_probs, 1)
            next_token = torch.gather(sorted_idx, -1, next_idx)
            x = torch.cat([x, next_token], dim=1)
        return x


m = TinyLM().to(device)
m.load_state_dict(ckpt["state_dict"])
m.eval()

history = []
max_turns = 6

print("Stinker chat. Ctrl+C to quit. Commands: /reset, /history")
while True:
    q = input("\nYou: ").strip()
    if not q:
        continue
    if q.lower() == "/reset":
        history = []
        print("Stinker: context reset.")
        continue
    if q.lower() == "/history":
        print("\n--- history ---")
        for i, (u, a) in enumerate(history, 1):
            print(f"{i}. You: {u}\n   Stinker: {a}")
        continue

    # Build a short rolling dialogue window.
    turns = history[-max_turns:]
    prompt_parts = []
    for u, a in turns:
        prompt_parts.append(f"{USER_TAG} {u}\n{ASSIST_TAG} {a} {END_TAG}\n")
    prompt_parts.append(f"{USER_TAG} {q}\n{ASSIST_TAG} ")
    prompt = "".join(prompt_parts)

    x = enc(prompt).unsqueeze(0)
    if x.numel() == 0:
        print("Stinker: I don't know these characters yet.")
        continue

    y = m.generate(x, n=200, temp=0.75, top_p=0.9, repetition_penalty=1.08)
    full = dec(y[0])
    reply = full[len(prompt) :]

    # stop on control tokens/new user turn
    stop_markers = [END_TAG, USER_TAG, "\n"]
    cut = len(reply)
    for marker in stop_markers:
        i = reply.find(marker)
        if i != -1:
            cut = min(cut, i)
    reply = reply[:cut].strip() or "..."

    history.append((q, reply))
    print("Stinker:", reply)

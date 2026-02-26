import platform

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_PATH = "gpt_stinker.ckpt"
TOKENIZER_PATH = "tokenizer.model"

device = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
ckpt = torch.load(CKPT_PATH, map_location=device)
cfg = ckpt["config"]

CONTEXT = cfg["context"]
N_LAYER = cfg["n_layer"]
N_HEAD = cfg["n_head"]
N_EMBD = cfg["n_embd"]
DROPOUT = 0.0
V = cfg["vocab"]

USER_TAG = "<|user|>"
ASSIST_TAG = "<|assistant|>"
END_TAG = "<|end|>"


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

    def forward(self, idx):
        _, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        return self.head(self.ln(self.blocks(x)))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=180, temp=0.75, top_p=0.9):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -CONTEXT:])[:, -1, :] / max(temp, 1e-4)
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


model = GPT().to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

if device == "cuda" and platform.system() != "Windows":
    try:
        model = torch.compile(model)
    except Exception:
        pass

history = []
max_turns = 6

print("GPT Stinker chat. Commands: /reset, /history, /quit")
while True:
    q = input("\nYou: ").strip()
    if not q:
        continue
    if q.lower() == "/quit":
        break
    if q.lower() == "/reset":
        history = []
        print("Stinker: context reset")
        continue
    if q.lower() == "/history":
        for i, (u, a) in enumerate(history, 1):
            print(f"{i}. You: {u}\n   Stinker: {a}")
        continue

    parts = []
    for u, a in history[-max_turns:]:
        parts.append(f"{USER_TAG} {u}\n{ASSIST_TAG} {a} {END_TAG}\n")
    parts.append(f"{USER_TAG} {q}\n{ASSIST_TAG} ")
    prompt = "".join(parts)

    ids = torch.tensor(sp.encode(prompt, out_type=int), dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(ids, max_new_tokens=180, temp=0.75, top_p=0.9)
    full = sp.decode(out[0].tolist())

    reply = full[len(prompt):]
    cut = len(reply)
    for marker in (END_TAG, USER_TAG, "\n"):
        i = reply.find(marker)
        if i != -1:
            cut = min(cut, i)
    reply = reply[:cut].strip() or "..."

    history.append((q, reply))
    print("Stinker:", reply)

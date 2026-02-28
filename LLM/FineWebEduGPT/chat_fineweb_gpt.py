import argparse
import platform

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, t, c = x.size()
        x_norm = self.ln1(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        if x.is_cuda:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            m = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=m,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.resid_drop(self.proj(y))
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
        self.head.weight = self.tok.weight

    def forward(self, idx):
        _, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        return self.head(self.ln(self.blocks(x)))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=220, temp=0.8, top_p=0.9):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.context :])[:, -1, :] / max(temp, 1e-4)
            probs = F.softmax(logits, dim=-1)
            s_probs, s_idx = torch.sort(probs, descending=True)
            csum = torch.cumsum(s_probs, dim=-1)
            mask = csum > top_p
            mask[..., 0] = False
            s_probs[mask] = 0
            s_probs = s_probs / s_probs.sum(dim=-1, keepdim=True)
            nxt = torch.multinomial(s_probs, 1)
            tok = torch.gather(s_idx, -1, nxt)
            idx = torch.cat([idx, tok], dim=1)
        return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="fineweb_gpt.ckpt")
    p.add_argument("--tok", default="tokenizer.model")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=args.tok)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["args"]

    model = GPT(
        vocab=ckpt["vocab"],
        context=cfg["context"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if device == "cuda" and platform.system() != "Windows":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    print("FineWebEduGPT chat. Commands: /quit")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q.lower() == "/quit":
            break
        prompt = f"User: {q}\nAssistant:"
        ids = torch.tensor(sp.encode(prompt, out_type=int), dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(ids, max_new_tokens=180, temp=0.75, top_p=0.9)
        text = sp.decode(out[0].tolist())
        reply = text[len(prompt):].split("\n")[0].strip()
        print("Assistant:", reply if reply else "...")


if __name__ == "__main__":
    main()

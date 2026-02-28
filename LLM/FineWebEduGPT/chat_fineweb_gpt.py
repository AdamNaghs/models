"""
Chat interface for FineWebEduGPT.

Works with both pretrained and finetuned (SFT) checkpoints.
- Pretrained: raw completion mode (no chat format)
- Finetuned: multi-turn chat with ### User: / ### Assistant: format

Usage:
    python chat_fineweb_gpt.py --ckpt fineweb_gpt_chat.ckpt --tok tokenizer.model
    python chat_fineweb_gpt.py --ckpt fineweb_gpt.ckpt --raw  # pretrained, no chat format
"""

import argparse
import platform

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model (must match training script exactly)
# ---------------------------------------------------------------------------

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
    def generate(self, idx, max_new_tokens=256, temp=0.7, top_p=0.9,
                 stop_tokens=None):
        """Generate tokens with top-p sampling and optional stop tokens."""
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.context:])[:, -1, :] / max(temp, 1e-4)
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
            # Stop on EOS or stop tokens.
            if stop_tokens and tok.item() in stop_tokens:
                break
        return idx


# Chat format constants (must match finetune_chat.py).
USER_PREFIX = "### User:\n"
ASST_PREFIX = "### Assistant:\n"
TURN_SUFFIX = "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="fineweb_gpt_chat.ckpt")
    p.add_argument("--tok", default="tokenizer.model")
    p.add_argument("--raw", action="store_true",
                   help="Raw completion mode (no chat format, for pretrained checkpoints)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-history", type=int, default=5,
                   help="Max conversation turns to keep in context")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=args.tok)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["args"]

    # Detect if this is a finetuned checkpoint.
    is_finetuned = "chat_format" in ckpt
    if args.raw:
        is_finetuned = False

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

    # EOS token for stopping generation.
    eos_id = sp.eos_id()
    # Also stop on the "###" token sequence (start of next turn).
    hash_ids = set(sp.encode("###", out_type=int))
    stop_tokens = {eos_id} | hash_ids if eos_id >= 0 else hash_ids

    mode_str = "chat (finetuned)" if is_finetuned else "completion (pretrained)"
    print(f"FineWebEduGPT {mode_str} | ctx={cfg['context']} | device={device}")
    print("Commands: /quit, /clear, /raw, /chat")
    print()

    history = []  # list of {"role": ..., "content": ...}

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            continue
        if q.lower() == "/quit":
            break
        if q.lower() == "/clear":
            history.clear()
            print("[history cleared]")
            continue
        if q.lower() == "/raw":
            is_finetuned = False
            print("[switched to raw completion mode]")
            continue
        if q.lower() == "/chat":
            is_finetuned = True
            print("[switched to chat mode]")
            continue

        if is_finetuned:
            # Build multi-turn prompt from history.
            history.append({"role": "user", "content": q})

            # Keep only recent turns to fit in context.
            recent = history[-(args.max_history * 2):]

            prompt = ""
            for msg in recent:
                if msg["role"] == "user":
                    prompt += USER_PREFIX + msg["content"] + TURN_SUFFIX
                elif msg["role"] == "assistant":
                    prompt += ASST_PREFIX + msg["content"] + TURN_SUFFIX
            prompt += ASST_PREFIX

            ids = sp.encode(prompt, out_type=int)
            # Truncate from the left if too long (keep recent context).
            max_prompt = cfg["context"] - args.max_tokens
            if len(ids) > max_prompt:
                ids = ids[-max_prompt:]

            idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            out = model.generate(
                idx, max_new_tokens=args.max_tokens,
                temp=args.temp, top_p=args.top_p,
                stop_tokens=stop_tokens,
            )

            generated = out[0].tolist()[len(ids):]
            reply = sp.decode(generated).strip()

            # Clean up: remove trailing ### or partial markers.
            for marker in ["### User:", "### Assistant:", "###"]:
                if marker in reply:
                    reply = reply[:reply.index(marker)].strip()

            # Remove EOS artifacts.
            reply = reply.replace("</s>", "").strip()

            if reply:
                history.append({"role": "assistant", "content": reply})
                print(f"Assistant: {reply}")
            else:
                print("Assistant: ...")
                history.pop()  # remove the unanswered user turn

        else:
            # Raw completion mode.
            prompt = f"User: {q}\nAssistant:"
            ids = torch.tensor(
                sp.encode(prompt, out_type=int), dtype=torch.long, device=device
            ).unsqueeze(0)
            out = model.generate(
                ids, max_new_tokens=args.max_tokens,
                temp=args.temp, top_p=args.top_p,
                stop_tokens=stop_tokens,
            )
            text = sp.decode(out[0].tolist())
            reply = text[len(prompt):].split("\n")[0].strip()
            print(f"Assistant: {reply if reply else '...'}")

        print()


if __name__ == "__main__":
    main()

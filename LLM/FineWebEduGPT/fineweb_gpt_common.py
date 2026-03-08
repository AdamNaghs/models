"""Shared model and artifact utilities for FineWebEduGPT."""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


USER_PREFIX = "### User:\n"
ASST_PREFIX = "### Assistant:\n"
TURN_SUFFIX = "\n"


class Block(nn.Module):
    """Single transformer block with pre-norm attention and FFN."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
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
        self._cpu_keep_mask_cache: dict[int, torch.Tensor] = {}

    def _cpu_keep_mask(self, seq_len: int, device: torch.device | str) -> torch.Tensor:
        mask = self._cpu_keep_mask_cache.get(seq_len)
        if mask is None or mask.device != torch.device(device):
            mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
            self._cpu_keep_mask_cache[seq_len] = mask
        return mask

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
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=self._cpu_keep_mask(t, x.device),
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.resid_drop(self.proj(y))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    """Decoder-only GPT model with tied input/output embeddings."""

    def __init__(self, vocab, context, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(context, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx, targets=None):
        _, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = (
            F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if targets is not None
            else None
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=256, temp=0.7, top_p=0.9,
                 stop_tokens=None, stop_sequences=None):
        """Generate tokens with top-p sampling and optional stop tokens/sequences."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.context:])
            logits = logits[:, -1, :] / max(temp, 1e-4)
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

            if stop_tokens and tok.item() in stop_tokens:
                break

            if stop_sequences:
                tail = idx[0].tolist()
                for seq in stop_sequences:
                    if seq and len(tail) >= len(seq) and tail[-len(seq):] == seq:
                        return idx
        return idx


def tokenizer_fingerprint(sp):
    """Stable tokenizer fingerprint to catch tokenizer/checkpoint drift."""
    sample = "\n".join(sp.id_to_piece(i) for i in range(min(sp.vocab_size(), 4096)))
    raw = f"vocab={sp.vocab_size()}|bos={sp.bos_id()}|eos={sp.eos_id()}|pad={sp.pad_id()}|{sample}"
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def unwrap_model(model):
    """Strip DDP/torch.compile wrappers for save/load-safe state_dict access."""
    raw = model
    while hasattr(raw, "module"):
        raw = raw.module
    while hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def resolve_tokenizer_path(ckpt_path: str, tok_path: str | None = None) -> str:
    """Default tokenizer to the checkpoint directory when not explicitly set."""
    if tok_path:
        return tok_path
    return os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "tokenizer.model")


def resolve_chat_output_path(ckpt_path: str, output_path: str | None = None) -> str:
    """Default finetuned checkpoint to the pretrained checkpoint directory."""
    if output_path:
        return output_path
    return os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "fineweb_gpt_chat.ckpt")


def resolve_step_output_path(output_path: str, step: int) -> str:
    """Place periodic finetune checkpoints next to the main output checkpoint."""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    return os.path.join(out_dir, f"fineweb_gpt_chat_step{step}.ckpt")

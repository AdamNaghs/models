#!/usr/bin/env python3
"""Fast sanity checks for FineWebEduGPT core paths.

Runs a tiny local tokenizer build, chat sample tokenization, one forward pass,
one masked-loss computation, one short generation, and a checkpoint round-trip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import sentencepiece as spm
import torch

from fineweb_gpt_common import ASST_PREFIX, GPT, USER_PREFIX, tokenizer_fingerprint, unwrap_model
from finetune_chat import masked_cross_entropy, tokenize_conversation_with_mask


def build_tiny_tokenizer(workdir: Path) -> spm.SentencePieceProcessor:
    corpus = workdir / "tiny_corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "Hello world",
                "The capital of France is Paris",
                "Transformers predict the next token",
                "FineWebEduGPT is a decoder only model",
                "User asks questions and Assistant answers",
            ]
            * 50
        ),
        encoding="utf-8",
    )
    prefix = workdir / "tiny_tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=64,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_id=0,
    )
    return spm.SentencePieceProcessor(model_file=str(prefix) + ".model")


def main():
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        sp = build_tiny_tokenizer(workdir)
        context = 32

        sample = [
            {"role": "user", "content": "What is two plus two?"},
            {"role": "assistant", "content": "Two plus two is four."},
        ]
        inp, tgt, mask = tokenize_conversation_with_mask(sample, sp, context=context, pad_id=sp.pad_id())
        assert inp is not None and tgt is not None and mask is not None, "chat tokenization failed"
        assert mask.sum().item() > 0, "assistant loss mask is empty"

        model = GPT(vocab=sp.vocab_size(), context=context, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
        logits, ce_loss = model(inp.unsqueeze(0), tgt.unsqueeze(0))
        assert logits.shape == (1, context, sp.vocab_size()), f"unexpected logits shape: {logits.shape}"
        assert ce_loss is not None and torch.isfinite(ce_loss), "forward CE loss is invalid"

        masked_loss = masked_cross_entropy(logits, tgt.unsqueeze(0), mask.unsqueeze(0))
        assert torch.isfinite(masked_loss), "masked loss is invalid"

        prompt_ids = sp.encode(USER_PREFIX + "Hello\n" + ASST_PREFIX, out_type=int)
        out = model.generate(torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0), max_new_tokens=4, temp=1.0, top_p=1.0)
        assert out.size(1) >= len(prompt_ids), "generation truncated unexpectedly"

        ckpt_path = workdir / "smoke.ckpt"
        ckpt = {
            "state_dict": unwrap_model(model).state_dict(),
            "args": {"context": context, "n_embd": 32, "n_head": 4, "n_layer": 2},
            "vocab": sp.vocab_size(),
            "tokenizer_fingerprint": tokenizer_fingerprint(sp),
        }
        torch.save(ckpt, ckpt_path)
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        reloaded = GPT(vocab=loaded["vocab"], context=context, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
        reloaded.load_state_dict(loaded["state_dict"])

    print("smoke_test.py: OK")


if __name__ == "__main__":
    main()

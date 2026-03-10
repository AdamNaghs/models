"""
Chat interface for FineWebEduGPT.

Works with both pretrained and finetuned (SFT) checkpoints.
- Pretrained: raw completion mode (no chat format)
- Finetuned: multi-turn chat with ### User: / ### Assistant: format

Usage:
    python chat_fineweb_gpt.py --ckpt runs/350m/fineweb_gpt_chat.ckpt
    python chat_fineweb_gpt.py --ckpt runs/350m/fineweb_gpt.ckpt --raw
"""

import argparse
import platform

import sentencepiece as spm
import torch

from fineweb_gpt_common import (
    ASST_PREFIX,
    GPT,
    TURN_SUFFIX,
    USER_PREFIX,
    resolve_tokenizer_path,
    tokenizer_fingerprint,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="fineweb_gpt_chat.ckpt")
    p.add_argument("--tok", default=None,
                   help="SentencePiece tokenizer model (defaults to <ckpt_dir>/tokenizer.model)")
    p.add_argument("--raw", action="store_true",
                   help="Raw completion mode (no chat format, for pretrained checkpoints)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-history", type=int, default=5,
                   help="Max conversation turns to keep in context")
    args = p.parse_args()
    args.tok = resolve_tokenizer_path(args.ckpt, args.tok)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=args.tok)
    tok_fp = tokenizer_fingerprint(sp)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["args"]

    ckpt_tok_fp = ckpt.get("tokenizer_fingerprint")
    if ckpt_tok_fp and ckpt_tok_fp != tok_fp:
        raise ValueError(
            "Tokenizer mismatch: tokenizer.model does not match checkpoint tokenizer. "
            "Use the same tokenizer from training/finetuning."
        )

    # Detect if this is a finetuned checkpoint.
    is_finetuned = "chat_format" in ckpt
    if args.raw:
        is_finetuned = False
    chat_format = ckpt.get(
        "chat_format",
        {"user_prefix": USER_PREFIX, "asst_prefix": ASST_PREFIX, "turn_suffix": TURN_SUFFIX},
    )
    user_prefix = chat_format["user_prefix"]
    asst_prefix = chat_format["asst_prefix"]
    turn_suffix = chat_format["turn_suffix"]

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

    # Stop on EOS token and full delimiter sequences.
    eos_id = sp.eos_id()
    stop_tokens = {eos_id} if eos_id >= 0 else set()
    stop_sequences = [
        sp.encode(user_prefix.strip(), out_type=int),
        sp.encode(asst_prefix.strip(), out_type=int),
    ]

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
                    prompt += user_prefix + msg["content"] + turn_suffix
                elif msg["role"] == "assistant":
                    prompt += asst_prefix + msg["content"] + turn_suffix
            prompt += asst_prefix

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
                stop_sequences=stop_sequences,
            )

            generated = out[0].tolist()[len(ids):]
            reply = sp.decode(generated).strip()

            # Clean up: remove trailing ### or partial markers.
            for marker in [user_prefix.strip(), asst_prefix.strip(), "###"]:
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
            prompt = q
            prompt_ids = sp.encode(prompt, out_type=int)
            idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            out = model.generate(
                idx, max_new_tokens=args.max_tokens,
                temp=args.temp, top_p=args.top_p,
                stop_tokens=stop_tokens,
                stop_sequences=stop_sequences,
            )
            generated = out[0].tolist()[len(prompt_ids):]
            reply = sp.decode(generated).strip()
            for marker in [user_prefix.strip(), asst_prefix.strip(), "###"]:
                if marker and marker in reply:
                    reply = reply[:reply.index(marker)].strip()
            reply = reply.replace("</s>", "").strip()
            print(f"Assistant: {reply if reply else '...'}")

        print()


if __name__ == "__main__":
    main()

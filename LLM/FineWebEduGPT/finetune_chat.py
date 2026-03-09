"""
Supervised finetuning for FineWebEduGPT on chat/instruction data.

Loads a pretrained checkpoint, trains on multi-turn conversations with
masked loss (only assistant tokens contribute to the loss), and saves
a chat-ready checkpoint.

Usage:
    python finetune_chat.py --ckpt runs/350m/fineweb_gpt.ckpt
    python finetune_chat.py --ckpt runs/350m/fineweb_gpt.ckpt --epochs 3 --lr 2e-5
    python finetune_chat.py --ckpt runs/350m/fineweb_gpt.ckpt --dataset custom --data-path my_data.jsonl
"""

import argparse
import json
import math
import os
import platform
import random
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from fineweb_gpt_common import (
    ASST_PREFIX,
    GPT,
    TURN_SUFFIX,
    USER_PREFIX,
    resolve_chat_output_path,
    resolve_step_output_path,
    resolve_tokenizer_path,
    tokenizer_fingerprint,
    unwrap_model,
)


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

def tokenize_conversation_with_mask(messages, sp, context, pad_id=3):
    """Tokenize a conversation and build an exact token-level assistant mask.

    This avoids fragile character-level alignment heuristics.
    """
    ids = []
    token_mask = []
    eos_id = sp.eos_id()

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content:
            continue

        if role == "user":
            part = sp.encode(USER_PREFIX + content + TURN_SUFFIX, out_type=int)
            ids.extend(part)
            token_mask.extend([0] * len(part))
        elif role == "assistant":
            prefix = sp.encode(ASST_PREFIX, out_type=int)
            body = sp.encode(content + TURN_SUFFIX, out_type=int)

            ids.extend(prefix)
            token_mask.extend([0] * len(prefix))

            ids.extend(body)
            token_mask.extend([1] * len(body))

            if eos_id >= 0:
                ids.append(eos_id)
                token_mask.append(1)

    if len(ids) < 2 or sum(token_mask) == 0:
        return None, None, None

    # Truncate to context + 1 (need one extra for shifted targets).
    max_len = context + 1
    if len(ids) > max_len:
        ids = ids[:max_len]
        token_mask = token_mask[:max_len]

    # Pad if shorter.
    pad_len = max_len - len(ids)
    ids = ids + [pad_id] * pad_len
    token_mask = token_mask + [0] * pad_len

    input_ids = torch.tensor(ids[:-1], dtype=torch.long)
    target_ids = torch.tensor(ids[1:], dtype=torch.long)
    loss_mask = torch.tensor(token_mask[1:], dtype=torch.long)

    if loss_mask.sum().item() == 0:
        return None, None, None

    return input_ids, target_ids, loss_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChatDataset(Dataset):
    """PyTorch Dataset for chat finetuning."""

    def __init__(self, conversations, sp, context, pad_id=3):
        self.sp = sp
        self.context = context
        self.pad_id = pad_id
        # Pre-process all conversations.
        self.samples = []
        skipped = 0
        for conv in conversations:
            result = tokenize_conversation_with_mask(conv, sp, context, pad_id)
            if result[0] is not None:
                self.samples.append(result)
            else:
                skipped += 1

        print(f"ChatDataset: {len(self.samples)} samples loaded, {skipped} skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_ultrachat(sp, context, max_samples=None, pad_id=3):
    """Load HuggingFaceH4/ultrachat_200k and return train/val ChatDatasets."""
    from datasets import load_dataset

    print("Loading ultrachat_200k from HuggingFace...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k")

    train_convs = []
    for ex in ds["train_sft"]:
        msgs = ex.get("messages", [])
        if msgs:
            if max_samples and len(train_convs) >= max_samples:
                break
            train_convs.append(msgs)

    val_convs = []
    for ex in ds["test_sft"]:
        msgs = ex.get("messages", [])
        if msgs:
            val_convs.append(msgs)
            if len(val_convs) >= 1000:
                break

    print(f"ultrachat: {len(train_convs)} train, {len(val_convs)} val conversations")

    train_ds = ChatDataset(train_convs, sp, context, pad_id)
    val_ds = ChatDataset(val_convs, sp, context, pad_id)
    return train_ds, val_ds


def load_custom_jsonl(path, sp, context, val_split=0.05, pad_id=3):
    """Load a custom JSONL file.

    Expected format per line:
        {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
    or:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
    """
    print(f"Loading custom data from {path}...")
    convs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("conversations") or obj.get("messages") or []
            if msgs:
                convs.append(msgs)

    random.shuffle(convs)
    split_idx = max(1, int(len(convs) * (1 - val_split)))
    train_convs = convs[:split_idx]
    val_convs = convs[split_idx:]

    print(f"custom: {len(train_convs)} train, {len(val_convs)} val conversations")

    train_ds = ChatDataset(train_convs, sp, context, pad_id)
    val_ds = ChatDataset(val_convs, sp, context, pad_id)
    return train_ds, val_ds


def masked_cross_entropy(logits, targets, loss_mask):
    """Compute token loss only where loss_mask == 1."""
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    mask_flat = loss_mask.view(-1).float()
    per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    return (per_token_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SFT finetuning for FineWebEduGPT")
    p.add_argument("--ckpt", required=True, help="Path to pretrained checkpoint")
    p.add_argument("--tok", default=None,
                   help="SentencePiece tokenizer model (defaults to <ckpt_dir>/tokenizer.model)")
    p.add_argument("--output", default=None,
                   help="Output checkpoint path (defaults to <ckpt_dir>/fineweb_gpt_chat.ckpt)")

    # Dataset
    p.add_argument("--dataset", default="ultrachat", choices=["ultrachat", "custom"],
                   help="Dataset to finetune on")
    p.add_argument("--data-path", default=None, help="Path to custom JSONL (when --dataset custom)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max training samples (None = use all)")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03,
                   help="Fraction of total steps used for warmup")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.05,
                   help="Dropout during finetuning (lower than pretraining)")
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # Logging / saving
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    args.tok = resolve_tokenizer_path(args.ckpt, args.tok)
    args.output = resolve_chat_output_path(args.ckpt, args.output)
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else "cpu"

    if is_cuda:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load tokenizer.
    sp = spm.SentencePieceProcessor(model_file=args.tok)
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else 3
    current_tok_fp = tokenizer_fingerprint(sp)

    # Load pretrained checkpoint.
    print(f"Loading pretrained checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_tok_fp = ckpt.get("tokenizer_fingerprint")
    if ckpt_tok_fp and ckpt_tok_fp != current_tok_fp:
        raise ValueError(
            "Tokenizer mismatch: finetune tokenizer does not match pretraining checkpoint tokenizer. "
            "Use the exact tokenizer.model used during pretraining."
        )
    cfg = ckpt["args"]
    context = cfg["context"]

    model = GPT(
        vocab=ckpt["vocab"],
        context=context,
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=args.dropout,
    ).to(device)

    # Load pretrained weights.
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"Loaded pretrained model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Architecture: {cfg['n_layer']}L / {cfg['n_head']}H / {cfg['n_embd']}E / ctx={context}")

    # Load dataset.
    if args.dataset == "ultrachat":
        train_ds, val_ds = load_ultrachat(sp, context, max_samples=args.max_samples, pad_id=pad_id)
    elif args.dataset == "custom":
        if not args.data_path:
            raise ValueError("--data-path is required when --dataset custom")
        train_ds, val_ds = load_custom_jsonl(args.data_path, sp, context, pad_id=pad_id)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if len(train_ds) == 0:
        raise RuntimeError("No training samples after processing. Check data format.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=is_cuda, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=is_cuda, drop_last=False,
    ) if len(val_ds) > 0 else None

    # Optimizer -- use lower LR than pretraining.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"\nFinetuning config:")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size} x grad_accum {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  steps/epoch: {steps_per_epoch}")
    print(f"  total_steps: {total_steps}")
    print(f"  warmup: {warmup_steps} steps")
    print(f"  lr: {args.lr} -> {args.min_lr}")
    print(f"  dropout: {args.dropout}")
    print(f"  device: {device}")
    print(f"  tokenizer: {args.tok}")
    print(f"  output: {args.output}")

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr + cosine * (args.lr - args.min_lr)

    @torch.no_grad()
    def eval_loss():
        if val_loader is None:
            return float("nan")
        model.eval()
        losses = []
        for i, (inp, tgt, mask) in enumerate(val_loader):
            if i >= args.eval_iters:
                break
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            with torch.amp.autocast("cuda", enabled=is_cuda):
                logits, _ = model(inp)
                loss = masked_cross_entropy(logits, tgt, mask)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses) if losses else float("nan")

    def save_ckpt(path, step, epoch, best_val=None):
        raw_model = unwrap_model(model)
        data = {
            "state_dict": raw_model.state_dict(),
            "args": cfg,  # preserve original architecture config
            "vocab": ckpt["vocab"],
            "step": step,
            "epoch": epoch,
            "finetune_args": vars(args),
            "chat_format": {
                "user_prefix": USER_PREFIX,
                "asst_prefix": ASST_PREFIX,
                "turn_suffix": TURN_SUFFIX,
            },
            "tokenizer_fingerprint": current_tok_fp,
        }
        if best_val is not None:
            data["best_val_loss"] = best_val
        tmp = path + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)

    # Compile if possible.
    if is_cuda and platform.system() != "Windows":
        print("Compiling model...")
        try:
            model = torch.compile(model)
        except Exception:
            print("torch.compile failed, continuing without")

    # Training loop.
    model.train()
    global_step = 0
    best_val = float("inf")
    best_saved = False
    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    print(f"\nStarting SFT training...\n")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        micro_step = 0
        accum_inflight = 0
        accum_target = None

        for batch_idx, (inp, tgt, mask) in enumerate(train_loader):
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            if accum_inflight == 0:
                accum_target = min(args.grad_accum, len(train_loader) - batch_idx)

            with torch.amp.autocast("cuda", enabled=is_cuda):
                logits, _ = model(inp)
                raw_loss = masked_cross_entropy(logits, tgt, mask)
                loss = raw_loss / accum_target

            scaler.scale(loss).backward()
            micro_step += 1
            accum_inflight += 1

            epoch_loss += raw_loss.item()
            epoch_tokens += mask.sum().item()

            if accum_inflight == accum_target:
                scaler.unscale_(optimizer)
                graded = [p for p in model.parameters() if p.grad is not None]
                if graded:
                    torch.nn.utils.clip_grad_norm_(graded, args.max_grad_norm)

                cur_lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum_inflight = 0
                accum_target = None

                # Logging.
                if args.log_every > 0 and global_step % args.log_every == 0:
                    avg = epoch_loss / max(batch_idx + 1, 1)
                    elapsed = time.perf_counter() - start_time
                    eta = (total_steps - global_step) * (elapsed / max(global_step, 1))
                    print(
                        f"epoch {epoch+1}/{args.epochs} | step {global_step}/{total_steps} | "
                        f"train {avg:.4f} | lr {cur_lr:.2e} | "
                        f"elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m"
                    )

                # Eval.
                if global_step % args.eval_every == 0:
                    val = eval_loss()
                    print(f"  -> val loss: {val:.4f} | ppl: {math.exp(val):.2f}")
                    if val < best_val:
                        best_val = val
                        best_saved = True
                        save_ckpt(args.output, global_step, epoch, best_val)
                        print(f"  -> new best! saved -> {args.output}")

                # Periodic checkpoint.
                if global_step % args.ckpt_every == 0:
                    ckpt_name = resolve_step_output_path(args.output, global_step)
                    save_ckpt(ckpt_name, global_step, epoch)
                    print(f"  -> checkpoint -> {ckpt_name}")

        # End of epoch eval.
        val = eval_loss()
        avg_epoch = epoch_loss / max(len(train_loader), 1)
        print(
            f"\n=== Epoch {epoch+1}/{args.epochs} complete | "
            f"train {avg_epoch:.4f} | val {val:.4f} | ppl {math.exp(val):.2f} ===\n"
        )
        if val < best_val:
            best_val = val
            best_saved = True
            save_ckpt(args.output, global_step, epoch, best_val)
            print(f"  -> new best! saved -> {args.output}")

    # Preserve args.output as the best checkpoint; only fall back to final
    # weights if no validation pass ever produced a saved artifact.
    if not best_saved:
        save_ckpt(args.output, global_step, args.epochs - 1, best_val if math.isfinite(best_val) else None)
    elapsed = time.perf_counter() - start_time
    print(f"\nDone. Total time: {elapsed/60:.1f}m")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()

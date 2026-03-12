"""
FineWebEduGPT Training Script
==============================
Trains a GPT-class language model on FineWeb-Edu.

Supported data loading modes:
  1. Local staged parquet (--offline --local-data-dir PATH): trains from
     manually staged chunks of parquet files with no network access.
  2. Local (default): loads the selected dataset config into the local
     Hugging Face cache and trains from Arrow-backed files.
"""

from __future__ import annotations

from datetime import timedelta

import math
import os
import platform
import random
import signal
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fineweb_gpt_common import GPT, tokenizer_fingerprint, unwrap_model
from fineweb_train_config import parse_args
from fineweb_training_data import SHUTDOWN_EVENT, ensure_tokenizer, make_batcher


def _signal_handler(signum, frame):
    """Set the shared shutdown event so the train loop exits cleanly."""
    del signum, frame
    SHUTDOWN_EVENT.set()


def estimate_params(vocab, context, n_embd, n_layer):
    """Approximate parameter count for a GPT with tied input/output embeddings."""
    return (
        vocab * n_embd
        + context * n_embd
        + n_layer * (12 * n_embd * n_embd + 9 * n_embd)
    )


def configure_runtime(seed):
    """Return rank/device information and initialize DDP when needed."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = world_size > 1
    is_cuda = torch.cuda.is_available()

    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    if is_cuda:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"

    if is_distributed:
        backend = "nccl" if is_cuda else "gloo"
        dist.init_process_group(
            backend=backend,
            device_id=torch.device(device) if backend == "nccl" else None,
            timeout=timedelta(hours=2),
        )

    return {
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "is_distributed": is_distributed,
        "is_cuda": is_cuda,
        "device": device,
        "is_main": rank == 0,
    }


def load_resume_checkpoint(args, device, model, current_tok_fp, is_main):
    """Restore model weights from a checkpoint if one was provided."""
    start_step = 0
    checkpoint = None
    if args.resume and os.path.exists(args.resume):
        if is_main:
            print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        ckpt_tok_fp = checkpoint.get("tokenizer_fingerprint")
        if ckpt_tok_fp and ckpt_tok_fp != current_tok_fp:
            raise RuntimeError(
                "Tokenizer mismatch while resuming checkpoint. "
                "Use the exact tokenizer.model from the original training run."
            )
        model.load_state_dict(checkpoint["state_dict"])
        start_step = checkpoint.get("step", 0) + 1
        if is_main:
            print(f"Resumed at step {start_step}")
    return start_step, checkpoint


def maybe_compile_model(model, is_cuda, no_compile, is_main):
    """Compile the model when CUDA is available and the user allows it."""
    if is_cuda and not no_compile and platform.system() != "Windows":
        if is_main:
            print("Compiling model with torch.compile (this takes 60-120s on first run)...")
        try:
            return torch.compile(model)
        except Exception:
            if is_main:
                print("torch.compile failed, continuing without compilation")
            return model
    if is_main and no_compile:
        print("Skipping torch.compile (--no-compile)")
    return model


def build_optimizer(model, args, is_cuda, is_main):
    """Build AdamW, preferring fused or foreach variants when available."""
    if is_cuda:
        try:
            if is_main:
                print("Optimizing with AdamW fused")
            return torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=args.weight_decay,
                fused=True,
            )
        except (TypeError, RuntimeError) as exc:
            if is_main:
                print(f"Fused AdamW unavailable ({exc}), falling back to foreach")
            try:
                return torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    betas=(0.9, 0.95),
                    weight_decay=args.weight_decay,
                    foreach=True,
                )
            except (TypeError, RuntimeError):
                if is_main:
                    print("foreach AdamW also unavailable, using vanilla AdamW")

    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )


def restore_training_state(checkpoint, optimizer, scaler, is_main):
    """Restore optimizer and scaler state after the checkpointed model loads."""
    if checkpoint is None:
        return
    if "opt_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt_state_dict"])
        if is_main:
            print("Restored optimizer state")
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if is_main:
            print("Restored scaler state")


def make_lr_schedule(args):
    """Linear warmup followed by cosine decay to args.min_lr."""
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.train_steps - args.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr + cosine * (args.lr - args.min_lr)

    return get_lr


def save_checkpoint(path, step, raw_model, optimizer, scaler, args, vocab, tokenizer_fingerprint_value):
    """Atomically write a resumable training checkpoint."""
    checkpoint = {
        "state_dict": raw_model.state_dict(),
        "opt_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "vocab": vocab,
        "step": step,
        "tokenizer_fingerprint": tokenizer_fingerprint_value,
    }
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def should_run_eval(step, start_step, args):
    """Skip the initial eval for offline staged runs, keep periodic evals unchanged."""
    if step == start_step and args.local_data_dir:
        return False
    if step == start_step:
        return True
    return step % args.eval_every == 0


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    args = parse_args()
    if args.offline and not args.local_data_dir:
        raise ValueError("--offline requires --local-data-dir pointing at staged parquet files.")

    runtime = configure_runtime(args.seed)
    device = runtime["device"]
    is_cuda = runtime["is_cuda"]
    is_distributed = runtime["is_distributed"]
    is_main = runtime["is_main"]
    local_rank = runtime["local_rank"]
    rank = runtime["rank"]
    world_size = runtime["world_size"]

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"out_dir: {args.out_dir}")
    if is_distributed:
        dist.barrier()

    tokenizer = ensure_tokenizer(args, is_main=is_main)
    vocab = tokenizer.vocab_size()
    current_tok_fp = tokenizer_fingerprint(tokenizer)

    train_batcher = make_batcher(
        tokenizer,
        args,
        rank=rank,
        world_size=world_size,
        is_main=is_main,
        is_val=False,
    )
    val_batcher = make_batcher(
        tokenizer,
        args,
        rank=rank,
        world_size=world_size,
        is_main=is_main,
        is_val=True,
    )

    model = GPT(
        vocab,
        args.context,
        args.n_embd,
        args.n_head,
        args.n_layer,
        args.dropout,
    ).to(device)

    start_step, checkpoint = load_resume_checkpoint(args, device, model, current_tok_fp, is_main)
    model = maybe_compile_model(model, is_cuda, args.no_compile, is_main)

    if is_distributed:
        ddp_device_ids = [local_rank] if is_cuda else None
        model = DDP(model, device_ids=ddp_device_ids)

    optimizer = build_optimizer(model, args, is_cuda, is_main)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
    restore_training_state(checkpoint, optimizer, scaler, is_main)
    checkpoint = None

    get_lr = make_lr_schedule(args)

    @torch.no_grad()
    def eval_loss(iters):
        model.eval()
        losses = []
        for _ in range(iters):
            xb, yb = val_batcher.next(device)
            with torch.amp.autocast("cuda", enabled=is_cuda):
                _, loss = model(xb, yb)
            losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)
        if is_distributed:
            reduced = torch.tensor([mean_loss], device=device, dtype=torch.float32)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            mean_loss = (reduced / world_size).item()
        model.train()
        return mean_loss

    raw_model = unwrap_model(model)
    params = sum(param.numel() for param in raw_model.parameters())
    est = estimate_params(vocab, args.context, args.n_embd, args.n_layer)
    tokens_per_step = args.batch_size * args.context * args.grad_accum * world_size
    data_mode = "local-staged" if args.local_data_dir else "local"

    if is_main:
        print(
            f"gpus={world_size} | device={device} | preset={args.preset or 'custom'} | "
            f"vocab={vocab} | params={params:,} (est {est:,}) | config={args.config} | "
            f"grad_accum={args.grad_accum} | workers={args.num_workers} | "
            f"global_batch={args.batch_size * args.grad_accum * world_size} | "
            f"data={data_mode} | seed={args.seed}"
        )
        if start_step > 0:
            print(f"Resuming from step {start_step}")
        if args.local_data_dir:
            print("Skipping initial eval for offline staged streaming")

    checkpoint_path = os.path.join(args.out_dir, "fineweb_gpt.ckpt")
    start_time = time.perf_counter()
    last_step_dt = 0.0
    last_data_wait_dt = 0.0
    last_compute_dt = 0.0
    last_opt_dt = 0.0
    last_other_dt = 0.0
    last_completed_step = start_step - 1
    final_checkpoint_saved = False
    train_loss_accum = 0.0
    train_loss_count = 0

    def no_sync_ctx():
        if is_distributed and isinstance(model, DDP):
            return model.no_sync()
        return nullcontext()

    try:
        for step in range(start_step, args.train_steps + 1):
            if SHUTDOWN_EVENT.is_set():
                if is_main:
                    print(f"Shutdown signal received at step {step}. Saving checkpoint...")
                    save_checkpoint(
                        checkpoint_path,
                        step,
                        raw_model,
                        optimizer,
                        scaler,
                        args,
                        vocab,
                        current_tok_fp,
                    )
                    print(f"checkpoint -> {checkpoint_path} (shutdown)")
                break

            now = time.perf_counter()
            elapsed = now - start_time
            steps_done = max(step - start_step, 1)
            avg_step = elapsed / steps_done
            eta = max(args.train_steps - step, 0) * avg_step

            if args.stop_after_one_epoch and step > start_step and hasattr(train_batcher, "epochs_completed"):
                if train_batcher.epochs_completed >= 1:
                    if is_main:
                        print("data: completed one full pass over the staged local chunk")
                        save_checkpoint(
                            checkpoint_path,
                            max(last_completed_step, start_step),
                            raw_model,
                            optimizer,
                            scaler,
                            args,
                            vocab,
                            current_tok_fp,
                        )
                        print(f"checkpoint -> {checkpoint_path} (chunk complete)")
                        print(
                            "Chunk training finished. Run download_fineweb_snapshot.py again to stage the next chunk for this sample, "
                            "then resubmit with --resume."
                        )
                    final_checkpoint_saved = True
                    if dist.is_initialized():
                        dist.barrier()
                    break

            if should_run_eval(step, start_step, args):
                if is_cuda:
                    torch.cuda.synchronize()
                eval_start = time.perf_counter()
                val_loss = eval_loss(args.eval_iters)
                eval_dt = time.perf_counter() - eval_start
                cur_lr = optimizer.param_groups[0]["lr"]
                toks_per_s = (tokens_per_step / last_step_dt) if last_step_dt > 0 else 0.0
                if is_main:
                    print(
                        f"step {step:5d} | val {val_loss:.4f} | ppl {math.exp(val_loss):.2f} | "
                        f"lr {cur_lr:.2e} | dt {last_step_dt:.2f}s | "
                        f"data {last_data_wait_dt:.2f}s | compute {last_compute_dt:.2f}s | "
                        f"opt {last_opt_dt:.2f}s | other {last_other_dt:.2f}s | "
                        f"tok/s {toks_per_s:,.0f} | eval {eval_dt:.2f}s | "
                        f"elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m"
                    )
                train_loss_accum = 0.0
                train_loss_count = 0

            step_start = time.perf_counter()
            cur_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr
            optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            data_wait_dt = 0.0
            compute_dt = 0.0
            compute_event_pairs = []

            for micro_idx in range(args.grad_accum):
                data_start = time.perf_counter()
                xb, yb = train_batcher.next(device)
                data_wait_dt += time.perf_counter() - data_start

                sync_context = no_sync_ctx() if micro_idx < args.grad_accum - 1 else nullcontext()
                if is_cuda:
                    compute_start = torch.cuda.Event(enable_timing=True)
                    compute_end = torch.cuda.Event(enable_timing=True)
                    compute_start.record()
                else:
                    compute_start = time.perf_counter()

                with sync_context:
                    with torch.amp.autocast("cuda", enabled=is_cuda):
                        _, loss = model(xb, yb)
                        loss = loss / args.grad_accum
                    scaler.scale(loss).backward()

                if is_cuda:
                    compute_end.record()
                    compute_event_pairs.append((compute_start, compute_end))
                else:
                    compute_dt += time.perf_counter() - compute_start
                step_loss += loss.item()

            if is_cuda:
                opt_start = torch.cuda.Event(enable_timing=True)
                opt_end = torch.cuda.Event(enable_timing=True)
                opt_start.record()
            else:
                opt_start = time.perf_counter()

            scaler.unscale_(optimizer)
            graded_params = [param for param in model.parameters() if param.grad is not None]
            if graded_params:
                torch.nn.utils.clip_grad_norm_(graded_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            if is_cuda:
                opt_end.record()
                torch.cuda.synchronize()
                compute_dt = sum(start.elapsed_time(end) for start, end in compute_event_pairs) / 1000.0
                opt_dt = opt_start.elapsed_time(opt_end) / 1000.0
            else:
                opt_dt = time.perf_counter() - opt_start

            last_step_dt = time.perf_counter() - step_start
            last_data_wait_dt = data_wait_dt
            last_compute_dt = compute_dt
            last_opt_dt = opt_dt
            last_other_dt = max(last_step_dt - last_data_wait_dt - last_compute_dt - last_opt_dt, 0.0)
            last_completed_step = step

            train_loss_accum += step_loss
            train_loss_count += 1

            if args.log_every > 0 and step > 0 and step % args.log_every == 0 and step % args.eval_every != 0:
                avg_train_loss = train_loss_accum / max(train_loss_count, 1)
                toks_per_s = (tokens_per_step / last_step_dt) if last_step_dt > 0 else 0.0
                if is_main:
                    print(
                        f"step {step:5d} | train {avg_train_loss:.4f} | "
                        f"lr {cur_lr:.2e} | dt {last_step_dt:.2f}s | "
                        f"data {last_data_wait_dt:.2f}s | compute {last_compute_dt:.2f}s | "
                        f"opt {last_opt_dt:.2f}s | other {last_other_dt:.2f}s | "
                        f"tok/s {toks_per_s:,.0f}"
                    )

            if step > 0 and step % args.ckpt_every == 0 and is_main:
                save_checkpoint(
                    checkpoint_path,
                    step,
                    raw_model,
                    optimizer,
                    scaler,
                    args,
                    vocab,
                    current_tok_fp,
                )
                print(f"checkpoint -> {checkpoint_path} | elapsed {(time.perf_counter() - start_time) / 60:.1f}m")

        if is_main and not SHUTDOWN_EVENT.is_set() and not final_checkpoint_saved:
            save_checkpoint(
                checkpoint_path,
                max(last_completed_step, start_step),
                raw_model,
                optimizer,
                scaler,
                args,
                vocab,
                current_tok_fp,
            )
            print(f"saved -> {checkpoint_path}")
    finally:
        train_batcher.close()
        val_batcher.close()
        if is_distributed and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
        if is_distributed and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()

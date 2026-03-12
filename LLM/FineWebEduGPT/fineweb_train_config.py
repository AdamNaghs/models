from __future__ import annotations

import argparse
import os
import sys


PRESETS = {
    "125m": {
        "train_steps": 20000,
        "batch_size": 8,
        "context": 2048,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "grad_accum": 16,
        "lr": 6e-4,
        "min_lr": 6e-5,
        "warmup_steps": 1000,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 200000,
    },
    "350m": {
        "train_steps": 30000,
        "batch_size": 32,
        "context": 2048,
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1024,
        "grad_accum": 4,
        "lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 2000,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 200000,
    },
    "760m": {
        "train_steps": 40000,
        "batch_size": 16,
        "context": 2048,
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1536,
        "grad_accum": 8,
        "lr": 2.5e-4,
        "min_lr": 2.5e-5,
        "warmup_steps": 2000,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 200000,
    },
    "1.3b": {
        "train_steps": 50000,
        "batch_size": 8,
        "context": 2048,
        "n_layer": 29,
        "n_head": 16,
        "n_embd": 1856,
        "grad_accum": 16,
        "lr": 2e-4,
        "min_lr": 2e-5,
        "warmup_steps": 3000,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 200000,
    },
}

SAMPLE_CONFIGS = (
    "sample-10BT",
    "sample-100BT",
    "sample-350BT",
)

DEFAULT_STORAGE_ROOT = os.environ.get("FINEWEB_STORAGE_ROOT", "/fs1/proj/educational_web_data")


def default_out_dir(preset: str | None) -> str:
    """Default artifact directory under the shared project storage root."""
    stage = preset if preset else "custom"
    return os.path.join(DEFAULT_STORAGE_ROOT, "runs", stage)


def resolve_local_data_dir(config: str) -> str:
    """Default staged-data directory for offline/manual chunk training."""
    return os.path.join(DEFAULT_STORAGE_ROOT, "dataset", "fineweb-edu", config, "source")


def _resolve_preset_aliases(args):
    if args.preset_125m:
        args.preset = "125m"
    elif args.preset_350m:
        args.preset = "350m"
    elif args.preset_760m:
        args.preset = "760m"
    elif args.preset_1_3b:
        args.preset = "1.3b"


def _apply_preset_overrides(args):
    explicit_flags = set(a.split("=")[0] for a in sys.argv[1:] if a.startswith("-"))
    key_to_flags = {
        "train_steps": {"--train-steps"},
        "batch_size": {"--batch-size"},
        "context": {"--context"},
        "n_layer": {"--n-layer"},
        "n_head": {"--n-head"},
        "n_embd": {"--n-embd"},
        "dropout": {"--dropout"},
        "lr": {"--lr"},
        "weight_decay": {"--weight-decay"},
        "eval_every": {"--eval-every"},
        "eval_iters": {"--eval-iters"},
        "ckpt_every": {"--ckpt-every"},
        "grad_accum": {"--grad-accum"},
        "warmup_steps": {"--warmup-steps"},
        "min_lr": {"--min-lr"},
        "vocab_size": {"--vocab-size"},
        "tokenizer_model": {"--tokenizer-model"},
        "queue_size": {"--queue-size"},
        "num_workers": {"--num-workers"},
        "seed_docs": {"--seed-docs"},
        "log_every": {"--log-every"},
        "local_data_dir": {"--local-data-dir"},
    }

    if args.preset:
        for key, value in PRESETS[args.preset].items():
            if not (key_to_flags.get(key, set()) & explicit_flags):
                setattr(args, key, value)

    if args.out_dir is None:
        args.out_dir = default_out_dir(args.preset)

    if "--tokenizer-model" not in explicit_flags:
        args.tokenizer_model = os.path.join(args.out_dir, "tokenizer.model")

    if args.local_data_dir is not None:
        args.local_data_dir = [os.path.abspath(path) for path in args.local_data_dir]


def parse_args():
    """Parse CLI arguments and apply preset overrides."""
    parser = argparse.ArgumentParser(description="GPT trainer on FineWeb-Edu")

    parser.add_argument(
        "--config",
        default="sample-100BT",
        choices=list(SAMPLE_CONFIGS),
        help="FineWeb-Edu dataset config",
    )

    parser.add_argument("--train-steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size per GPU")
    parser.add_argument("--context", type=int, default=512, help="Sequence length (context window)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps (linear)")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum LR for cosine schedule")

    parser.add_argument("--n-layer", type=int, default=12, help="Number of transformer blocks")
    parser.add_argument("--n-head", type=int, default=10, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=640, help="Embedding dimension")
    parser.add_argument("--vocab-size", type=int, default=16000, help="SentencePiece vocabulary size")

    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--eval-iters", type=int, default=100, help="Batches per evaluation")
    parser.add_argument("--ckpt-every", type=int, default=2000, help="Checkpoint every N steps")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log training loss every N steps (0 = disabled)",
    )

    parser.add_argument(
        "--tokenizer-model",
        default="tokenizer.model",
        help="Path to SentencePiece .model file (auto-built if missing)",
    )
    parser.add_argument(
        "--seed-docs",
        type=int,
        default=50000,
        help="Number of docs streamed to build tokenizer if missing",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        default=False,
        help="Disable torch.compile (useful when compile-time memory overhead is too high)",
    )

    parser.add_argument("--queue-size", type=int, default=64, help="Prefetch queue size for batchers")
    parser.add_argument("--num-workers", type=int, default=2, help="Background tokenization threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument(
        "--preset",
        choices=["125m", "350m", "760m", "1.3b"],
        help="Apply a size-based training preset (override any field via CLI flags)",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Disable HuggingFace network access and use only staged local parquet",
    )
    parser.add_argument(
        "--local-data-dir",
        action="append",
        default=None,
        help="Directory containing staged parquet files for offline/manual chunk training. Repeat to combine multiple configs.",
    )
    parser.add_argument(
        "--stop-after-one-epoch",
        action="store_true",
        default=False,
        help="Stop after one full pass over the current local dataset chunk",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and tokenizer",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")

    aliases = parser.add_mutually_exclusive_group()
    aliases.add_argument("-125M", dest="preset_125m", action="store_true")
    aliases.add_argument("-350M", dest="preset_350m", action="store_true")
    aliases.add_argument("-760M", dest="preset_760m", action="store_true")
    aliases.add_argument("-1.3B", dest="preset_1_3b", action="store_true")

    args = parser.parse_args()
    _resolve_preset_aliases(args)
    _apply_preset_overrides(args)
    return args

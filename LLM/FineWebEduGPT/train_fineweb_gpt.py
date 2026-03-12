"""
FineWebEduGPT Training Script
==============================
Trains a GPT-class language model on the HuggingFace FineWeb-Edu dataset.

Supported data loading modes:
  1. Local staged parquet (--offline --local-data-dir PATH): Trains from
     manually staged chunks of parquet files with no network access.
  2. Local (default): Loads the selected dataset config into HF's local cache
     and trains from Arrow-backed files.

Multi-GPU training via PyTorch DDP (torchrun --nproc_per_node=N).

Usage examples:
  # Offline staged chunk from shared storage
  torchrun --nproc_per_node=8 train_fineweb_gpt.py -125M --offline \
      --local-data-dir /fs1/proj/educational_web_data/dataset/fineweb-edu/sample-10BT/source \
      --out-dir /fs1/proj/educational_web_data/runs/125m \
      --stop-after-one-epoch

  # Offline across a staged sample chunk
  torchrun --nproc_per_node=8 train_fineweb_gpt.py -125M --offline \
      --local-data-dir /fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source \
      --out-dir /fs1/proj/educational_web_data/runs/125m

  # Resume from checkpoint
  torchrun --nproc_per_node=8 train_fineweb_gpt.py -125M --offline \
      --local-data-dir /fs1/proj/educational_web_data/dataset/fineweb-edu/sample-100BT/source \
      --resume /fs1/proj/educational_web_data/runs/125m/fineweb_gpt.ckpt \
      --out-dir /fs1/proj/educational_web_data/runs/125m
"""

from __future__ import annotations

from datetime import timedelta

import argparse
import glob
import itertools
import math
import os
import platform
import queue
import random
import signal
import sys
import threading
import time
import tempfile
from collections import deque
from contextlib import nullcontext

import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset

from fineweb_gpt_common import GPT, tokenizer_fingerprint, unwrap_model


# =============================================================================
# Hardware Presets
# =============================================================================
# Each preset configures model architecture and training hyperparameters
# optimized for a specific GPU tier. Override individual params via CLI flags.

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

# HuggingFace dataset identifier. FineWeb-Edu is large across all configs.
# We never download the full dataset -- only individual dataset configs or
# staged local parquet chunks.
HF_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_STORAGE_ROOT = os.environ.get("FINEWEB_STORAGE_ROOT", "/fs1/proj/educational_web_data")

# Global shutdown event for graceful SIGTERM/SIGINT handling.
# Worker threads check this periodically to drain cleanly.
_shutdown = threading.Event()


def _signal_handler(signum, frame):
    """Set shutdown event on SIGTERM/SIGINT so worker threads drain cleanly."""
    _shutdown.set()


def default_out_dir(preset: str | None) -> str:
    """Default artifact directory under the shared project storage root."""
    stage = preset if preset else "custom"
    return os.path.join(DEFAULT_STORAGE_ROOT, "runs", stage)


def resolve_local_data_dir(config: str) -> str:
    """Default staged-data directory for offline/manual chunk training."""
    return os.path.join(DEFAULT_STORAGE_ROOT, "dataset", "fineweb-edu", config, "source")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse CLI arguments and apply preset overrides.

    Presets set defaults for architecture and training params, but any
    explicitly passed CLI flag takes priority over the preset value.
    """
    p = argparse.ArgumentParser(description="GPT trainer on FineWeb-Edu")

    # Dataset config: which FineWeb-Edu dataset config to train on.
    p.add_argument(
        "--config",
        default="sample-100BT",
        choices=list(SAMPLE_CONFIGS),
        help="FineWeb-Edu dataset config",
    )

    # Training hyperparameters.
    p.add_argument("--train-steps", type=int, default=100000, help="Total training steps")
    p.add_argument("--batch-size", type=int, default=8, help="Micro-batch size per GPU")
    p.add_argument("--context", type=int, default=512, help="Sequence length (context window)")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    p.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
    p.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    p.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps (linear)")
    p.add_argument("--min-lr", type=float, default=3e-5, help="Minimum LR for cosine schedule")

    # Model architecture.
    p.add_argument("--n-layer", type=int, default=12, help="Number of transformer blocks")
    p.add_argument("--n-head", type=int, default=10, help="Number of attention heads")
    p.add_argument("--n-embd", type=int, default=640, help="Embedding dimension")
    p.add_argument("--vocab-size", type=int, default=16000, help="SentencePiece vocabulary size")

    # Evaluation and checkpointing.
    p.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps")
    p.add_argument("--eval-iters", type=int, default=100, help="Batches per evaluation")
    p.add_argument("--ckpt-every", type=int, default=2000, help="Checkpoint every N steps")
    p.add_argument("--log-every", type=int, default=10,
                   help="Log training loss every N steps (0 = disabled)")

    # Tokenizer.
    p.add_argument("--tokenizer-model", default="tokenizer.model",
                   help="Path to SentencePiece .model file (auto-built if missing)")
    p.add_argument("--seed-docs", type=int, default=50000,
                   help="Number of docs streamed to build tokenizer if missing")
    p.add_argument("--no-compile", action="store_true", default=False,
                   help="Disable torch.compile (useful when compile-time memory overhead is too high)")

    # Data loading infrastructure.
    p.add_argument("--queue-size", type=int, default=64, help="Prefetch queue size for batchers")
    p.add_argument("--num-workers", type=int, default=2, help="Background tokenization threads")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Hardware preset shortcuts.
    p.add_argument("--preset", choices=["125m", "350m", "760m", "1.3b"],
                   help="Apply a size-based training preset (tuned for multi-GPU runs; override any field via CLI flags)")

    p.add_argument("--offline", action="store_true", default=False,
                   help="Disable HuggingFace network access and use only staged local parquet")
    p.add_argument("--local-data-dir", action="append", default=None,
                   help="Directory containing staged parquet files for offline/manual chunk training. Repeat to combine multiple configs.")
    p.add_argument("--stop-after-one-epoch", action="store_true", default=False,
                   help="Stop after one full pass over the current local dataset chunk")

    # Output directory.
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory for checkpoints and tokenizer "
                        "(default: <storage-root>/runs/<preset> or <storage-root>/runs/custom)")

    # Checkpoint resume.
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint file to resume training from")

    # Shorthand flags for presets: -125M, -350M, -760M, -1.3B
    g = p.add_mutually_exclusive_group()
    g.add_argument("-125M", dest="preset_125m", action="store_true")
    g.add_argument("-350M", dest="preset_350m", action="store_true")
    g.add_argument("-760M", dest="preset_760m", action="store_true")
    g.add_argument("-1.3B", dest="preset_1_3b", action="store_true")

    args = p.parse_args()

    # Resolve shorthand preset flags.
    if args.preset_125m:
        args.preset = "125m"
    elif args.preset_350m:
        args.preset = "350m"
    elif args.preset_760m:
        args.preset = "760m"
    elif args.preset_1_3b:
        args.preset = "1.3b"

    # Apply preset values, but only for params the user didn't explicitly set.
    # We detect explicit flags by checking sys.argv against the known flag names.
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
        for k, v in PRESETS[args.preset].items():
            if not (key_to_flags.get(k, set()) & explicit_flags):
                setattr(args, k, v)

    # Resolve output directory (after preset so we know the preset name).
    if args.out_dir is None:
        args.out_dir = default_out_dir(args.preset)

    # Default tokenizer path inside out_dir (unless user explicitly set it).
    if "--tokenizer-model" not in explicit_flags:
        args.tokenizer_model = os.path.join(args.out_dir, "tokenizer.model")

    if args.local_data_dir is not None:
        args.local_data_dir = [os.path.abspath(path) for path in args.local_data_dir]

    return args


# =============================================================================
# Utilities
# =============================================================================

def estimate_params(vocab, context, n_embd, n_layer):
    """Estimate total model parameters (approximate).

    Per block: QKV projection (3*E^2) + output projection (E^2) +
    FFN (8*E^2 + 5*E) + LayerNorm (4*E) = ~12*E^2 + 9*E per block.
    Plus token embedding (V*E) and positional embedding (T*E).
    Output head shares weights with token embedding (weight tying).
    """
    return (
        vocab * n_embd
        + context * n_embd
        + n_layer * (12 * n_embd * n_embd + 9 * n_embd)
    )


def format_local_data_dirs(local_data_dirs):
    if not local_data_dirs:
        return "(unset)"
    if isinstance(local_data_dirs, str):
        local_data_dirs = [local_data_dirs]
    return ", ".join(os.path.abspath(path) for path in local_data_dirs)


def discover_local_parquet_files(local_data_dir, *, required=False):
    """Recursively discover staged parquet files under one or more source trees."""
    if not local_data_dir:
        if required:
            raise RuntimeError("Local parquet data is required but --local-data-dir was not provided.")
        return []

    roots = [local_data_dir] if isinstance(local_data_dir, str) else list(local_data_dir)
    parquet_files = []
    missing_roots = []
    for root in roots:
        abs_root = os.path.abspath(root)
        if not os.path.exists(abs_root):
            missing_roots.append(abs_root)
            continue
        parquet_files.extend(glob.glob(os.path.join(abs_root, "**", "*.parquet"), recursive=True))
    parquet_files = sorted(set(parquet_files))
    if required and not parquet_files:
        raise RuntimeError(
            f"No parquet files found under local data dirs: {format_local_data_dirs(roots)}\n"
            "Run download_fineweb_snapshot.py first to stage the next chunk."
        )
    if required and missing_roots:
        raise RuntimeError(
            f"Missing local data dirs: {', '.join(missing_roots)}\n"
            "Run download_fineweb_snapshot.py first to stage the requested configs."
        )
    return parquet_files


def build_parquet_work_items(local_data_dir):
    """Enumerate parquet row-group work items for direct staged-data streaming."""
    parquet_files = discover_local_parquet_files(local_data_dir, required=True)
    work_items = []
    for parquet_path in parquet_files:
        pf = pq.ParquetFile(parquet_path)
        num_row_groups = pf.metadata.num_row_groups
        for row_group_idx in range(num_row_groups):
            rows = pf.metadata.row_group(row_group_idx).num_rows
            work_items.append(
                {
                    "path": parquet_path,
                    "row_group": row_group_idx,
                    "rows": rows,
                }
            )
    return parquet_files, work_items


def split_parquet_work_items(work_items):
    """Deterministically reserve a tail subset of work items for validation."""
    if not work_items:
        raise RuntimeError("No parquet work items discovered from staged local data.")

    total_items = len(work_items)
    if total_items == 1:
        raise RuntimeError(
            "Need at least 2 staged parquet work items to create disjoint train/val splits."
        )

    n_val = min(max(100, total_items // 100), total_items - 1)
    train_items = work_items[:-n_val]
    val_items = work_items[-n_val:]
    if not train_items or not val_items:
        raise RuntimeError(
            f"Unable to split staged parquet work items into train/val. total_items={total_items}, "
            f"train={len(train_items)}, val={len(val_items)}"
    )
    return train_items, val_items


def assign_work_items_by_rows(work_items, world_size):
    """Balance row-group work across ranks by cumulative row count."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    assignments = [[] for _ in range(world_size)]
    rank_row_totals = [0] * world_size
    ordered_items = sorted(
        work_items,
        key=lambda item: (-item["rows"], item["path"], item["row_group"]),
    )
    for item in ordered_items:
        target_rank = min(range(world_size), key=lambda idx: (rank_row_totals[idx], idx))
        assignments[target_rank].append(item)
        rank_row_totals[target_rank] += item["rows"]

    for rank_items in assignments:
        rank_items.sort(key=lambda item: (item["path"], item["row_group"]))
    return assignments


def summarize_row_assignments(assignments):
    """Return min/max row and item counts for per-rank work assignments."""
    if not assignments:
        return {
            "min_rows": 0,
            "max_rows": 0,
            "min_items": 0,
            "max_items": 0,
        }
    row_totals = [sum(item["rows"] for item in rank_items) for rank_items in assignments]
    item_totals = [len(rank_items) for rank_items in assignments]
    return {
        "min_rows": min(row_totals),
        "max_rows": max(row_totals),
        "min_items": min(item_totals),
        "max_items": max(item_totals),
    }


PARQUET_READ_BATCH_SIZE = 4096


def iter_text_from_work_item(work_item):
    """Yield text values from a single parquet row-group work item."""
    pf = pq.ParquetFile(work_item["path"])
    batch_iter = pf.iter_batches(
        columns=["text"],
        batch_size=PARQUET_READ_BATCH_SIZE,
        row_groups=[work_item["row_group"]],
        use_threads=True,
    )
    for batch in batch_iter:
        column = batch.column(0)
        if isinstance(column, pa.ChunkedArray):
            values = column.combine_chunks().to_pylist()
        else:
            values = column.to_pylist()
        for text in values:
            text = (text or "").strip()
            if text:
                yield text


def write_tokenizer_seed_from_parquet(seed_path, parquet_files, seed_docs):
    """Write tokenizer seed documents from staged local parquet."""
    written = 0
    with open(seed_path, "w", encoding="utf-8") as f:
        for parquet_path in parquet_files:
            pf = pq.ParquetFile(parquet_path)
            for batch in pf.iter_batches(columns=["text"], batch_size=PARQUET_READ_BATCH_SIZE, use_threads=True):
                texts = batch.column(0).to_pylist()
                for text in texts:
                    text = (text or "").strip()
                    if not text:
                        continue
                    f.write(text + "\n")
                    written += 1
                    if written >= seed_docs:
                        return written
    return written


def ensure_tokenizer(args, is_main=True):
    """Load or build a SentencePiece BPE tokenizer.

    If the tokenizer .model file exists, loads it directly. Otherwise,
    builds from staged local parquet.
    In distributed training, only rank 0 builds; other ranks wait at barrier.
    """
    tok_model = args.tokenizer_model

    # Handle Windows path differences for SentencePiece model prefix.
    if platform.system() == "Windows":
        prefix = tok_model.replace(".model", "")
    else:
        prefix = tok_model[:-6] if tok_model.endswith(".model") else tok_model

    # Try loading existing tokenizer.
    try:
        sp = spm.SentencePieceProcessor(model_file=tok_model)
        if is_main:
            print(f"Using existing tokenizer: {tok_model}")
        return sp
    except Exception:
        pass

    local_parquet_files = discover_local_parquet_files(args.local_data_dir, required=False)

    # Build tokenizer from local staged data (rank 0 only).
    if is_main:
        os.makedirs(os.path.dirname(os.path.abspath(tok_model)) or ".", exist_ok=True)
        fd, seed_path = tempfile.mkstemp(
            prefix="tokenizer_seed_",
            suffix=".txt",
            dir=os.path.dirname(os.path.abspath(tok_model)) or None,
            text=True,
        )
        os.close(fd)
        wrote_seed = 0
        try:
            if local_parquet_files:
                print(
                    f"Tokenizer missing. Building from first {args.seed_docs:,} docs "
                    f"in staged local parquet: {args.local_data_dir}"
                )
                wrote_seed = write_tokenizer_seed_from_parquet(seed_path, local_parquet_files, args.seed_docs)
            else:
                raise RuntimeError(
                    "Tokenizer missing and no staged parquet was provided.\n"
                    f"Expected tokenizer at: {tok_model}\n"
                    f"Expected staged parquet under: {format_local_data_dirs(args.local_data_dir or [resolve_local_data_dir(args.config)])}"
                )

            if wrote_seed == 0:
                raise RuntimeError(
                    "Unable to collect any tokenizer seed documents.\n"
                    f"Checked local data dirs: {format_local_data_dirs(args.local_data_dir)}"
                )

            spm.SentencePieceTrainer.train(
                input=seed_path,
                model_prefix=prefix,
                vocab_size=args.vocab_size,
                model_type="bpe",
                character_coverage=1.0,
                bos_id=1,
                eos_id=2,
                pad_id=3,
                unk_id=0,
            )
        finally:
            try:
                os.remove(seed_path)
            except OSError:
                pass

    # Wait for rank 0 to finish building before other ranks try to load.
    if dist.is_initialized():
        dist.barrier()

    return spm.SentencePieceProcessor(model_file=tok_model)


# =============================================================================
# Data Loading: Local Batcher
# =============================================================================

class _AtomicCounter:
    """Thread-safe counter for distributing document indices across workers."""

    def __init__(self, start=0):
        self._val = start
        self._lock = threading.Lock()

    def get_and_increment(self):
        with self._lock:
            v = self._val
            self._val += 1
            return v

    def reset(self, val=0):
        with self._lock:
            self._val = val


class LocalBatcher:
    """High-throughput batcher from a pre-loaded HF Dataset (Arrow-backed).

    Key properties:
    - Zero network I/O: reads from memory-mapped local Arrow files.
    - Disjoint per-rank sharding: each GPU processes non-overlapping docs.
    - Atomic counter: prevents document duplication across worker threads.
    - Epoch reshuffling: reshuffles document order each epoch with unique seed.
    - Graceful shutdown: respects global _shutdown event for SIGTERM handling.

    Worker threads tokenize documents and pack tokens into fixed-length
    sequences (context+1 tokens: context for input, +1 for shifted target).
    Sequences are enqueued for the training loop to consume.
    """

    def __init__(self, sp, dataset, context, batch_size, queue_size=256,
                 num_workers=4, rank=0, world_size=1, seed=42):
        self.sp = sp
        self.eos_id = sp.eos_id()
        self.context = context
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=queue_size)
        self.stop = threading.Event()
        self.rank = rank
        self.world_size = max(1, world_size)
        self.seed = seed

        self.dataset = dataset

        # Shard documents across ranks: rank 0 gets indices [0, W, 2W, ...],
        # rank 1 gets [1, W+1, 2W+1, ...], etc.
        total = len(dataset)
        self.indices = list(range(rank, total, world_size))

        self.num_workers = max(1, num_workers)
        self._counter = _AtomicCounter(0)
        self._epoch = 0
        self._epoch_count = 0
        self._epoch_lock = threading.Lock()
        self._epoch_id = 0  # Monotonic epoch id to prevent stale index usage.
        self._shuffled_indices = list(self.indices)
        random.Random(seed + rank).shuffle(self._shuffled_indices)

        # Launch background tokenization workers.
        self.workers = []
        for _ in range(self.num_workers):
            w = threading.Thread(target=self._run, daemon=True)
            w.start()
            self.workers.append(w)

    def _should_stop(self):
        return self.stop.is_set() or _shutdown.is_set()

    def _get_next_doc_index(self):
        """Returns (epoch_id, doc_index) or (epoch_id, None) if epoch exhausted."""
        epoch_id = self._epoch_id
        pos = self._counter.get_and_increment()
        if pos < len(self._shuffled_indices):
            return epoch_id, self._shuffled_indices[pos]
        return epoch_id, None

    def _advance_epoch(self, from_epoch_id):
        """Advance to next epoch. Only one thread actually advances; others see
        the updated epoch_id and skip."""
        with self._epoch_lock:
            if self._epoch_id != from_epoch_id:
                return  # Another thread already advanced.
            self._epoch += 1
            self._epoch_count += 1
            self._epoch_id += 1
            self._shuffled_indices = list(self.indices)
            random.Random(self.seed + self.rank + self._epoch * 1000).shuffle(self._shuffled_indices)
            self._counter.reset(0)

    @property
    def epochs_completed(self):
        return self._epoch_count

    def _run(self):
        """Worker thread: tokenize documents and pack into batches."""
        token_buf = deque()
        needed = self.batch_size * (self.context + 1)

        while not self._should_stop():
            epoch_id, idx = self._get_next_doc_index()
            if idx is None:
                self._advance_epoch(epoch_id)
                continue

            # Skip if epoch changed since we grabbed this index.
            if self._epoch_id != epoch_id:
                continue

            txt = (self.dataset[idx].get("text") or "").strip()
            if not txt:
                continue

            # Tokenize and append EOS to separate documents in the stream.
            ids = self.sp.encode(txt, out_type=int)
            if ids:
                token_buf.extend(ids)
                token_buf.append(self.eos_id)

            # When we have enough tokens, pack into batch and enqueue.
            while len(token_buf) >= needed and not self._should_stop():
                block = list(itertools.islice(token_buf, needed))
                for _ in range(needed):
                    token_buf.popleft()
                t = torch.tensor(block, dtype=torch.long).view(self.batch_size, self.context + 1)
                x = t[:, :-1].contiguous()   # Input: tokens [0..T-1]
                y = t[:, 1:].contiguous()     # Target: tokens [1..T]
                while not self._should_stop():
                    try:
                        self.q.put((x, y), timeout=1.0)
                        break
                    except queue.Full:
                        continue

    def next(self, device):
        """Get next batch, moving tensors to device with pinned memory if CUDA."""
        x, y = self.q.get(timeout=120)
        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def close(self):
        self.stop.set()


class LocalParquetStreamBatcher:
    """Stream staged parquet row groups directly without building an Arrow cache.

    Each rank receives a deterministic shard of the global row-group work list.
    Worker threads stream text batches from local parquet, tokenize on the fly,
    and pack fixed-length token blocks into the training queue.
    """

    def __init__(
        self,
        sp,
        assigned_items,
        context,
        batch_size,
        queue_size=256,
        num_workers=4,
        label="train",
        rank=0,
        seed=42,
        queue_timeout=120,
    ):
        self.sp = sp
        self.eos_id = sp.eos_id()
        self.context = context
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=queue_size)
        self.stop = threading.Event()
        self.label = label
        self.rank = rank
        self.seed = seed
        self.queue_timeout = queue_timeout

        self.assigned_items = list(assigned_items)
        self.assigned_item_count = len(self.assigned_items)
        self.assigned_rows = sum(item["rows"] for item in self.assigned_items)
        if not self.assigned_items:
            raise RuntimeError(
                f"Rank {rank} received no staged parquet work items for {label} loading."
            )
        self.num_workers = max(1, num_workers)
        self._counter = _AtomicCounter(0)
        self._epoch = 0
        self._epoch_count = 0
        self._epoch_lock = threading.Lock()
        self._epoch_id = 0
        self._shuffled_items = list(self.assigned_items)
        random.Random(seed + rank).shuffle(self._shuffled_items)

        self.workers = []
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._run, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _should_stop(self):
        return self.stop.is_set() or _shutdown.is_set()

    def _get_next_work_item(self):
        epoch_id = self._epoch_id
        pos = self._counter.get_and_increment()
        if pos < len(self._shuffled_items):
            return epoch_id, self._shuffled_items[pos]
        return epoch_id, None

    def _advance_epoch(self, from_epoch_id):
        with self._epoch_lock:
            if self._epoch_id != from_epoch_id:
                return
            self._epoch += 1
            self._epoch_count += 1
            self._epoch_id += 1
            self._shuffled_items = list(self.assigned_items)
            random.Random(self.seed + self.rank + self._epoch * 1000).shuffle(self._shuffled_items)
            self._counter.reset(0)

    @property
    def epochs_completed(self):
        return self._epoch_count

    def _run(self):
        token_buf = deque()
        needed = self.batch_size * (self.context + 1)

        while not self._should_stop():
            epoch_id, work_item = self._get_next_work_item()
            if work_item is None:
                self._advance_epoch(epoch_id)
                continue

            if self._epoch_id != epoch_id:
                continue

            for text in iter_text_from_work_item(work_item):
                if self._should_stop():
                    return

                ids = self.sp.encode(text, out_type=int)
                if not ids:
                    continue
                token_buf.extend(ids)
                token_buf.append(self.eos_id)

                while len(token_buf) >= needed and not self._should_stop():
                    block = list(itertools.islice(token_buf, needed))
                    for _ in range(needed):
                        token_buf.popleft()
                    t = torch.tensor(block, dtype=torch.long).view(self.batch_size, self.context + 1)
                    x = t[:, :-1].contiguous()
                    y = t[:, 1:].contiguous()
                    while not self._should_stop():
                        try:
                            self.q.put((x, y), timeout=1.0)
                            break
                        except queue.Full:
                            continue

    def next(self, device):
        try:
            x, y = self.q.get(timeout=self.queue_timeout)
        except queue.Empty as exc:
            raise RuntimeError(
                f"Timed out waiting for staged parquet {self.label} batch on rank {self.rank}. "
                f"assigned_items={self.assigned_item_count}, assigned_rows={self.assigned_rows}, "
                f"queue_timeout={self.queue_timeout}s"
            ) from exc
        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def close(self):
        self.stop.set()


# =============================================================================
# Batcher Factory
# =============================================================================

def make_batcher(sp, args, *, rank=0, world_size=1, is_main=False, is_val=False):
    """Create the appropriate batcher based on the data loading mode.

    Priority:
      1. --local-data-dir: staged local parquet (offline/manual chunks).
      2. Default: Load full config into HF local cache (fastest for networked environments).
    """
    if args.local_data_dir:
        qsize = max(16, args.queue_size // 2) if is_val else args.queue_size
        nw = max(1, args.num_workers // 2) if is_val else args.num_workers
        parquet_files = None
        train_work_items = None
        val_work_items = None
        train_assignments = None
        val_assignments = None
        if is_main:
            parquet_files, all_work_items = build_parquet_work_items(args.local_data_dir)
            train_work_items, val_work_items = split_parquet_work_items(all_work_items)
            train_assignments = assign_work_items_by_rows(train_work_items, world_size)
            val_assignments = assign_work_items_by_rows(val_work_items, world_size)
            train_summary = summarize_row_assignments(train_assignments)
            val_summary = summarize_row_assignments(val_assignments)
            print(
                f"data: streaming {len(parquet_files)} staged parquet files from "
                f"{format_local_data_dirs(args.local_data_dir)}"
            )
            print(
                f"data: direct parquet streaming enabled | work_items={len(all_work_items):,} | "
                f"train={len(train_work_items):,} | val={len(val_work_items):,}"
            )
            print(
                "data: row-balanced rank assignment | "
                f"train_rows_per_rank={train_summary['min_rows']:,}-{train_summary['max_rows']:,} | "
                f"val_rows_per_rank={val_summary['min_rows']:,}-{val_summary['max_rows']:,} | "
                f"train_items_per_rank={train_summary['min_items']}-{train_summary['max_items']} | "
                f"val_items_per_rank={val_summary['min_items']}-{val_summary['max_items']}"
            )

        work_meta = [parquet_files, train_work_items, val_work_items, train_assignments, val_assignments]
        if dist.is_initialized():
            gathered = [work_meta]
            dist.broadcast_object_list(gathered, src=0)
            parquet_files, train_work_items, val_work_items, train_assignments, val_assignments = gathered[0]

        if not is_main and not dist.is_initialized():
            parquet_files, all_work_items = build_parquet_work_items(args.local_data_dir)
            train_work_items, val_work_items = split_parquet_work_items(all_work_items)
            train_assignments = assign_work_items_by_rows(train_work_items, world_size)
            val_assignments = assign_work_items_by_rows(val_work_items, world_size)

        selected_items = val_work_items if is_val else train_work_items
        selected_label = "val" if is_val else "train"
        selected_assignments = val_assignments if is_val else train_assignments
        if len(selected_items) < world_size:
            raise RuntimeError(
                f"Not enough staged parquet work items for distributed {selected_label} loading. "
                f"items={len(selected_items)}, world_size={world_size}. Stage a larger chunk or reduce GPUs."
            )
        assigned_rank_items = selected_assignments[rank]
        if is_main:
            print(
                f"data({selected_label}): using {len(selected_items):,} staged parquet work items "
                f"via direct streaming"
            )
        return LocalParquetStreamBatcher(
            sp, assigned_rank_items, args.context, args.batch_size,
            queue_size=qsize, num_workers=nw,
            label=selected_label,
            rank=rank,
            seed=args.seed + 9999 if is_val else args.seed,
        )

    # Default: load the selected dataset config into HF's local Arrow cache.
    # This downloads only the selected FineWeb-Edu config, NOT the
    # entire dataset.
    qsize = max(16, args.queue_size // 2) if is_val else args.queue_size
    nw = max(1, args.num_workers // 2) if is_val else args.num_workers

    ds = None
    if is_main:
        print("data: loading dataset config from local cache (downloading config if needed)...")
        ds = load_dataset(HF_DATASET, args.config, split="train")
        print(f"data: {len(ds):,} documents loaded")

    if dist.is_initialized():
        dist.barrier()

    if ds is None:
        ds = load_dataset(HF_DATASET, args.config, split="train")

    # Split into train (99%) and val (1%) to avoid data overlap.
    n_val = max(100, len(ds) // 100)
    if is_val:
        val_ds = ds.select(range(len(ds) - n_val, len(ds)))
        if is_main:
            print(f"data(val): using {len(val_ds):,} held-out docs")
        return LocalBatcher(
            sp, val_ds, args.context, args.batch_size,
            queue_size=qsize, num_workers=nw,
            rank=rank, world_size=world_size,
            seed=args.seed + 9999,
        )
    else:
        train_ds = ds.select(range(len(ds) - n_val))
        if is_main:
            print(f"data(train): {len(train_ds):,} docs (reserved {n_val:,} for val)")
        return LocalBatcher(
            sp, train_ds, args.context, args.batch_size,
            queue_size=qsize, num_workers=nw,
            rank=rank, world_size=world_size,
            seed=args.seed,
        )


# =============================================================================
# Training Loop
# =============================================================================

def main():
    # Install signal handlers before any GPU/NCCL init for graceful shutdown.
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    args = parse_args()

    if args.offline and not args.local_data_dir:
        raise ValueError(
            "--offline requires --local-data-dir pointing at staged parquet files."
        )

    # --- Distributed training setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = world_size > 1
    is_cuda = torch.cuda.is_available()

    # Seed everything for reproducibility. Each rank gets a unique seed
    # derived from the base seed so they see different data orderings.
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    # --- Device setup ---
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

    is_main = (rank == 0)

    # --- Output directory ---
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"out_dir: {args.out_dir}")
    if is_distributed:
        dist.barrier()

    # --- Tokenizer ---
    sp = ensure_tokenizer(args, is_main=is_main)
    vocab = sp.vocab_size()
    current_tok_fp = tokenizer_fingerprint(sp)

    # --- Data loaders ---
    train_batcher = make_batcher(
        sp, args, rank=rank, world_size=world_size,
        is_main=is_main, is_val=False,
    )
    val_batcher = make_batcher(
        sp, args, rank=rank, world_size=world_size,
        is_main=is_main, is_val=True,
    )

    # --- Model ---
    model = GPT(
        vocab, args.context, args.n_embd, args.n_head,
        args.n_layer, args.dropout,
    ).to(device)

    # --- Resume from checkpoint ---
    start_step = 0
    _ckpt = None
    if args.resume and os.path.exists(args.resume):
        if is_main:
            print(f"Resuming from checkpoint: {args.resume}")
        _ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        ckpt_tok_fp = _ckpt.get("tokenizer_fingerprint")
        if ckpt_tok_fp and ckpt_tok_fp != current_tok_fp:
            raise RuntimeError(
                "Tokenizer mismatch while resuming checkpoint. "
                "Use the exact tokenizer.model from the original training run."
            )
        model.load_state_dict(_ckpt["state_dict"])
        start_step = _ckpt.get("step", 0) + 1
        if is_main:
            print(f"Resumed at step {start_step}")

    # --- torch.compile (skip on Windows due to triton issues) ---
    if is_cuda and not args.no_compile and platform.system() != "Windows":
        if is_main:
            print("Compiling model with torch.compile (this takes 60-120s on first run)...")
        try:
            model = torch.compile(model)
        except Exception:
            if is_main:
                print("torch.compile failed, continuing without compilation")
    elif is_main and args.no_compile:
        print("Skipping torch.compile (--no-compile)")

    # --- DDP wrapping ---
    if is_distributed:
        ddp_device_ids = [local_rank] if is_cuda else None
        model = DDP(model, device_ids=ddp_device_ids)

    # --- Optimizer ---
    # Try fused AdamW (fastest), then foreach, then vanilla as fallback.
    opt = None
    if is_cuda:
        try:
            if is_main:
                print("Optimizing with AdamW fused")
            opt = torch.optim.AdamW(
                model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                weight_decay=args.weight_decay, fused=True,
            )
        except (TypeError, RuntimeError) as e:
            if is_main:
                print(f"Fused AdamW unavailable ({e}), falling back to foreach")
            try:
                opt = torch.optim.AdamW(
                    model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                    weight_decay=args.weight_decay, foreach=True,
                )
            except (TypeError, RuntimeError):
                if is_main:
                    print("foreach AdamW also unavailable, using vanilla AdamW")

    if opt is None:
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

    # Mixed precision gradient scaler (no-op on CPU).
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    # Restore optimizer and scaler state from checkpoint.
    if _ckpt is not None:
        if "opt_state_dict" in _ckpt:
            opt.load_state_dict(_ckpt["opt_state_dict"])
            if is_main:
                print("Restored optimizer state")
        if "scaler_state_dict" in _ckpt:
            scaler.load_state_dict(_ckpt["scaler_state_dict"])
            if is_main:
                print("Restored scaler state")
        del _ckpt  # Free checkpoint memory.

    # --- Learning rate schedule ---
    # Linear warmup -> cosine decay to min_lr.
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.train_steps - args.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr + cosine * (args.lr - args.min_lr)

    # --- Evaluation ---
    @torch.no_grad()
    def eval_loss(iters):
        """Compute average validation loss across `iters` batches.
        In distributed mode, losses are averaged across all ranks."""
        model.eval()
        vals = []
        for _ in range(iters):
            xb, yb = val_batcher.next(device)
            with torch.amp.autocast("cuda", enabled=is_cuda):
                _, loss = model(xb, yb)
            vals.append(loss.item())
        local_mean = sum(vals) / len(vals)
        if is_distributed:
            t = torch.tensor([local_mean], device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            local_mean = (t / world_size).item()
        model.train()
        return local_mean

    # --- Training stats ---
    # Unwrap DDP and torch.compile to get the raw model for param counting.
    raw_model = unwrap_model(model)
    params = sum(p.numel() for p in raw_model.parameters())
    est = estimate_params(vocab, args.context, args.n_embd, args.n_layer)
    tokens_per_step = args.batch_size * args.context * args.grad_accum * world_size

    if args.local_data_dir:
        data_mode = "local-staged"
    else:
        data_mode = "local"

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

    # --- Checkpoint saving ---
    def save_checkpoint(path, step):
        """Atomic checkpoint save: write to .tmp then os.replace.

        Only called on rank 0. Saves model weights, optimizer state, scaler
        state, and all training args for full resumability.
        """
        ckpt_data = {
            "state_dict": raw_model.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "args": vars(args),
            "vocab": vocab,
            "step": step,
            "tokenizer_fingerprint": current_tok_fp,
        }
        tmp_path = path + ".tmp"
        torch.save(ckpt_data, tmp_path)
        os.replace(tmp_path, path)

    ckpt_path = os.path.join(args.out_dir, "fineweb_gpt.ckpt")
    start_time = time.perf_counter()
    last_step_dt = 0.0
    last_completed_step = start_step - 1
    final_checkpoint_saved = False

    # Accumulator for logging average training loss between eval steps.
    train_loss_accum = 0.0
    train_loss_count = 0

    # Context manager for skipping DDP gradient sync on non-final micro-steps.
    def no_sync_ctx():
        if is_distributed and isinstance(model, DDP):
            return model.no_sync()
        return nullcontext()

    # --- Main training loop ---
    try:
        for step in range(start_step, args.train_steps + 1):
            # Check for graceful shutdown (SIGTERM/SIGINT).
            if _shutdown.is_set():
                if is_main:
                    print(f"Shutdown signal received at step {step}. Saving checkpoint...")
                    save_checkpoint(ckpt_path, step)
                    print(f"checkpoint -> {ckpt_path} (shutdown)")
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
                        save_checkpoint(ckpt_path, max(last_completed_step, start_step))
                        print(f"checkpoint -> {ckpt_path} (chunk complete)")
                        print(
                            "Chunk training finished. Run download_fineweb_snapshot.py again to stage the next chunk for this sample, "
                            "then resubmit with --resume."
                        )
                    final_checkpoint_saved = True
                    if dist.is_initialized():
                        dist.barrier()
                    break

            # --- Periodic evaluation ---
            should_run_eval = (step % args.eval_every == 0)
            if step == start_step and args.local_data_dir:
                should_run_eval = False
            elif step == start_step:
                should_run_eval = True
            if should_run_eval:
                if is_cuda:
                    torch.cuda.synchronize()
                eval_start = time.perf_counter()
                v = eval_loss(args.eval_iters)
                eval_dt = time.perf_counter() - eval_start
                cur_lr = opt.param_groups[0]["lr"]
                toks_per_s = (tokens_per_step / last_step_dt) if last_step_dt > 0 else 0.0
                if is_main:
                    print(
                        f"step {step:5d} | val {v:.4f} | ppl {math.exp(v):.2f} | "
                        f"lr {cur_lr:.2e} | dt {last_step_dt:.2f}s | tok/s {toks_per_s:,.0f} | "
                        f"eval {eval_dt:.2f}s | elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m"
                    )
                train_loss_accum = 0.0
                train_loss_count = 0

            # --- Training step ---
            step_start = time.perf_counter()

            # Update learning rate (cosine schedule with warmup).
            cur_lr = get_lr(step)
            for param_group in opt.param_groups:
                param_group["lr"] = cur_lr
            opt.zero_grad(set_to_none=True)

            # Gradient accumulation: sum gradients over micro-steps.
            # Skip DDP all-reduce on non-final micro-steps for efficiency.
            step_loss = 0.0
            for micro_idx in range(args.grad_accum):
                xb, yb = train_batcher.next(device)
                ctx = no_sync_ctx() if micro_idx < args.grad_accum - 1 else nullcontext()
                with ctx:
                    with torch.amp.autocast("cuda", enabled=is_cuda):
                        _, loss = model(xb, yb)
                        loss = loss / args.grad_accum  # Normalize for accumulation.
                    scaler.scale(loss).backward()
                step_loss += loss.item()

            # Gradient clipping and optimizer step.
            scaler.unscale_(opt)
            graded_params = [p for p in model.parameters() if p.grad is not None]
            if graded_params:
                torch.nn.utils.clip_grad_norm_(graded_params, 1.0)
            scaler.step(opt)
            scaler.update()

            if is_cuda:
                torch.cuda.synchronize()
            last_step_dt = time.perf_counter() - step_start
            last_completed_step = step

            # Track training loss for periodic logging.
            train_loss_accum += step_loss
            train_loss_count += 1

            # --- Periodic training loss logging (between eval steps) ---
            if args.log_every > 0 and step > 0 and step % args.log_every == 0 and step % args.eval_every != 0:
                avg_train_loss = train_loss_accum / max(train_loss_count, 1)
                toks_per_s = (tokens_per_step / last_step_dt) if last_step_dt > 0 else 0.0
                if is_main:
                    print(
                        f"step {step:5d} | train {avg_train_loss:.4f} | "
                        f"lr {cur_lr:.2e} | dt {last_step_dt:.2f}s | tok/s {toks_per_s:,.0f}"
                    )

            # --- Periodic checkpointing ---
            if step > 0 and step % args.ckpt_every == 0 and is_main:
                save_checkpoint(ckpt_path, step)
                print(f"checkpoint -> {ckpt_path} | elapsed {(time.perf_counter()-start_time)/60:.1f}m")

        # Save final checkpoint at end of training.
        if is_main and not _shutdown.is_set() and not final_checkpoint_saved:
            save_checkpoint(ckpt_path, max(last_completed_step, start_step))
            print(f"saved -> {ckpt_path}")
    finally:
        # Clean shutdown: close batchers and destroy process group.
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

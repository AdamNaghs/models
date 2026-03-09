"""
FineWebEduGPT Training Script
==============================
Trains a GPT-class language model on the HuggingFace FineWeb-Edu dataset.

Supports three data loading modes:
  1. Rolling cache (--cache-gb N): Downloads N GB of shards, trains on them,
     deletes, and fetches the next chunk. Best for limited disk space.
  2. Streaming (--stream): Streams directly from HuggingFace Hub with zero
     local storage. Slowest but requires no disk.
  3. Local staged parquet (--offline --local-data-dir PATH): Trains from a
     manually staged chunk of parquet files with no network access.
  4. Local (default): Loads the dataset config into HF's local cache and
     trains from memory-mapped Arrow files. Fastest throughput.

Multi-GPU training via PyTorch DDP (torchrun --nproc_per_node=N).

Usage examples:
  # Single GPU, 350M preset, rolling cache
  python train_fineweb_gpt.py -350M --cache-gb 5 --out-dir runs/350m

  # Multi-GPU (8x H100), 1.3B preset, streaming mode
  torchrun --nproc_per_node=8 train_fineweb_gpt.py -1.3B --stream --out-dir runs/1.3b

  # Offline staged chunk from shared storage
  torchrun --nproc_per_node=8 train_fineweb_gpt.py -125M --offline \
      --local-data-dir /fs1/proj/educational_web_data/dataset/fineweb-edu/CC-MAIN-2025-26/source \
      --out-dir /fs1/proj/educational_web_data/runs/125m \
      --stop-after-one-epoch

  # Resume from checkpoint
  python train_fineweb_gpt.py -350M --cache-gb 5 --out-dir runs/350m --resume runs/350m/fineweb_gpt.ckpt
"""

from __future__ import annotations

from datetime import timedelta

import argparse
import glob
import itertools
import json
import math
import os
import platform
import queue
import random
import signal
import shutil
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
        "batch_size": 64,
        "context": 2048,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "grad_accum": 2,
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

# HuggingFace dataset identifier. FineWeb-Edu is ~10TB total across all
# configs. We never download the full dataset -- only individual configs
# (CommonCrawl snapshots) via streaming or rolling cache.
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


def default_cache_dir(config: str, preset: str | None) -> str:
    """Keep rolling-cache shards under the shared bulk-storage dataset tree."""
    stage = preset if preset else "custom"
    return os.path.join(DEFAULT_STORAGE_ROOT, "dataset", "fineweb-edu", config, stage)


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
    p = argparse.ArgumentParser(description="Full-streaming GPT trainer on FineWeb-Edu")

    # Dataset config: which CommonCrawl snapshot to train on.
    p.add_argument(
        "--config",
        default="CC-MAIN-2025-26",
        choices=[
            "CC-MAIN-2025-05",
            "CC-MAIN-2025-08",
            "CC-MAIN-2025-13",
            "CC-MAIN-2025-18",
            "CC-MAIN-2025-21",
            "CC-MAIN-2025-26",
        ],
        help="CommonCrawl snapshot config from FineWeb-Edu",
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

    # Data loading infrastructure.
    p.add_argument("--queue-size", type=int, default=64, help="Prefetch queue size for batchers")
    p.add_argument("--num-workers", type=int, default=2, help="Background tokenization threads")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Hardware preset shortcuts.
    p.add_argument("--preset", choices=["125m", "350m", "760m", "1.3b"],
                   help="Apply a size-based training preset (tuned for multi-GPU runs; override any field via CLI flags)")

    # Data loading modes (mutually exclusive in practice):
    #   --stream: Pure network streaming, no local storage needed.
    #   --cache-gb N: Rolling cache -- downloads N GB, trains, deletes, repeats.
    #   (default): Loads full config into HF cache. NOT full 10TB dataset --
    #              just the selected CommonCrawl snapshot config.
    p.add_argument("--stream", action="store_true", default=False,
                   help="Use HF streaming mode (slower, no pre-download needed)")
    p.add_argument("--cache-gb", type=float, default=0,
                   help="Rolling cache size in GB. Downloads this much data, trains on it, "
                        "deletes, and downloads the next chunk. (0 = use full config download)")
    p.add_argument("--cache-dir", type=str, default=".data_cache",
                   help="Directory for rolling cache parquet files")
    p.add_argument("--offline", action="store_true", default=False,
                   help="Disable HuggingFace network access and use only staged local parquet")
    p.add_argument("--local-data-dir", type=str, default=None,
                   help="Directory containing staged parquet files for offline/manual chunk training")
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

    # Default cache dir inside out_dir (unless user explicitly set it).
    if "--cache-dir" not in explicit_flags:
        args.cache_dir = default_cache_dir(args.config, args.preset)

    if args.local_data_dir is not None:
        args.local_data_dir = os.path.abspath(args.local_data_dir)

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


def discover_local_parquet_files(local_data_dir, *, required=False):
    """Recursively discover staged parquet files under a local source tree."""
    if not local_data_dir:
        if required:
            raise RuntimeError("Local parquet data is required but --local-data-dir was not provided.")
        return []

    root = os.path.abspath(local_data_dir)
    parquet_files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if required and not parquet_files:
        raise RuntimeError(
            f"No parquet files found under local data dir: {root}\n"
            "Run download_fineweb_snapshot.py first to stage the next chunk."
        )
    return parquet_files


def load_local_parquet_dataset(local_data_dir, *, is_main=False, label="data"):
    """Load a staged local parquet tree into an Arrow-backed HF Dataset."""
    parquet_files = discover_local_parquet_files(local_data_dir, required=True)
    if is_main:
        print(f"{label}: loading {len(parquet_files)} staged parquet files from {local_data_dir}...")

    from datasets import Dataset

    tables = [pq.read_table(path, columns=["text"]) for path in parquet_files]
    combined = pa.concat_tables(tables)
    ds = Dataset(combined)

    if is_main:
        print(f"{label}: {len(ds):,} documents loaded from staged local parquet")
    return ds


def write_tokenizer_seed_from_parquet(seed_path, parquet_files, seed_docs):
    """Write tokenizer seed documents from staged local parquet."""
    written = 0
    with open(seed_path, "w", encoding="utf-8") as f:
        for parquet_path in parquet_files:
            pf = pq.ParquetFile(parquet_path)
            for batch in pf.iter_batches(columns=["text"], batch_size=512):
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
    builds from staged local parquet or streams seed_docs documents from
    HuggingFace to build a new tokenizer.
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

    # Build tokenizer from local staged data or streamed data (rank 0 only).
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
            elif args.offline:
                raise RuntimeError(
                    "Tokenizer missing and offline mode is enabled.\n"
                    f"Expected tokenizer at: {tok_model}\n"
                    f"Expected staged parquet under: {args.local_data_dir or resolve_local_data_dir(args.config)}"
                )
            else:
                print(f"Tokenizer missing. Building from first {args.seed_docs:,} streamed docs...")
                ds = load_dataset(HF_DATASET, args.config, split="train", streaming=True)
                with open(seed_path, "w", encoding="utf-8") as f:
                    for ex in itertools.islice(ds, args.seed_docs):
                        t = (ex.get("text") or "").strip()
                        if t:
                            f.write(t + "\n")
                            wrote_seed += 1

            if wrote_seed == 0:
                raise RuntimeError(
                    "Unable to collect any tokenizer seed documents.\n"
                    f"Checked local data dir: {args.local_data_dir or '(unset)'}"
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


# =============================================================================
# Data Loading: Streaming Batcher
# =============================================================================

class _SharedDocIterator:
    """Thread-safe iterator over an HF streaming dataset.

    Multiple worker threads share a single dataset iterator. The lock
    ensures only one thread advances the iterator at a time.
    """

    def __init__(self, config, rank=0, world_size=1):
        ds = load_dataset(HF_DATASET, config, split="train", streaming=True)
        if world_size > 1:
            try:
                from datasets.distributed import split_dataset_by_node
                ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
            except ImportError:
                # Fallback: manual strided iteration.
                ds = itertools.islice(ds, rank, None, world_size)
        self._it = iter(ds)
        self._lock = threading.Lock()

    def __next__(self):
        with self._lock:
            return next(self._it)


class StreamingBatcher:
    """Network-streaming batcher for training without any local storage.

    Streams documents directly from HuggingFace Hub. Slower than local
    loading due to network latency, but requires zero disk space.
    """

    def __init__(self, sp, config, context, batch_size, queue_size=64,
                 num_workers=2, rank=0, world_size=1):
        self.sp = sp
        self.eos_id = sp.eos_id()
        self.context = context
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=queue_size)
        self.stop = threading.Event()
        self.num_workers = max(1, num_workers)
        self._shared_docs = _SharedDocIterator(config, rank=rank, world_size=world_size)

        self.workers = []
        for _ in range(self.num_workers):
            w = threading.Thread(target=self._run, daemon=True)
            w.start()
            self.workers.append(w)

    def _should_stop(self):
        return self.stop.is_set() or _shutdown.is_set()

    def _run(self):
        """Worker thread: stream, tokenize, and pack into batches."""
        token_buf = deque()
        needed = self.batch_size * (self.context + 1)

        while not self._should_stop():
            try:
                ex = next(self._shared_docs)
            except StopIteration:
                break  # Dataset exhausted.

            txt = (ex.get("text") or "").strip()
            if not txt:
                continue
            ids = self.sp.encode(txt, out_type=int)
            if ids:
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
        """Get next batch, moving to device."""
        x, y = self.q.get(timeout=120)
        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def close(self):
        self.stop.set()


# =============================================================================
# Data Loading: Rolling Cache
# =============================================================================
# Downloads N GB of parquet shards -> trains on them -> deletes -> downloads
# the next batch of shards. This way you never need the full dataset on disk.

def _list_parquet_urls(config):
    """List all parquet file URLs for a FineWeb-Edu config via HF Hub API.

    Uses HfApi.list_repo_tree() to discover shard files. We never guess
    shard names because HuggingFace naming conventions vary by dataset.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_tree(HF_DATASET, repo_type="dataset", path_in_repo=f"data/{config}")
        urls = []
        for f in files:
            if hasattr(f, "rfilename") and f.rfilename.endswith(".parquet"):
                urls.append(f.rfilename)
        urls.sort()
        if not urls:
            raise RuntimeError(
                f"No parquet files found for config '{config}' in {HF_DATASET}. "
                f"Check that the config name is correct and you have network access."
            )
        return urls
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for --cache-gb mode. "
            "Install with: pip install huggingface_hub"
        )


def _load_cache_state(state_path):
    """Load rolling cache progress state from disk."""
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        state.setdefault("current_shard_idx", 0)
        state.setdefault("next_shard_idx", 0)
        state.setdefault("total_shards_trained", 0)
        return state
    return {"current_shard_idx": 0, "next_shard_idx": 0, "total_shards_trained": 0}


def _save_cache_state(state_path, state):
    """Persist rolling cache progress (atomic write via tmp + replace)."""
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def _download_shard_batch(shard_paths, cache_dir, max_bytes, is_main=True):
    """Download parquet shards into cache_dir up to max_bytes total size."""
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)
    downloaded = []
    total_bytes = 0

    for shard_path in shard_paths:
        if total_bytes >= max_bytes:
            break
        try:
            local_path = hf_hub_download(
                repo_id=HF_DATASET,
                filename=shard_path,
                repo_type="dataset",
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )
            fsize = os.path.getsize(local_path)
            total_bytes += fsize
            downloaded.append(local_path)
            if is_main:
                print(f"  cached: {shard_path} ({fsize / 1e9:.2f} GB, total: {total_bytes / 1e9:.2f} GB)")
        except Exception as e:
            if is_main:
                print(f"  skip: {shard_path} ({e})")
            continue

    return downloaded


def _clear_cache_data(cache_dir, is_main=True):
    """Delete parquet data files but preserve metadata (state, shard list)."""
    preserve = {"_shard_list.json", "_chunk_count.json"}
    if not os.path.exists(cache_dir):
        return
    for entry in os.listdir(cache_dir):
        if entry in preserve:
            continue
        path = os.path.join(cache_dir, entry)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError as e:
            if is_main:
                print(f"  warning: could not remove {path}: {e}")
    if is_main:
        print(f"cache: cleared data in {cache_dir}")


class RollingCacheBatcher:
    """Downloads data in chunks, trains on each chunk, then rotates.

    Wraps LocalBatcher internally. The training loop calls should_rotate()
    which uses an all-reduce so all DDP ranks agree on when to rotate.

    Chunk lifecycle:
      1. Download N GB of parquet shards from HuggingFace Hub.
      2. Load into a LocalBatcher for high-throughput training.
      3. When all docs in the chunk have been seen (1 epoch), rotate:
         save checkpoint, delete old data, download next batch.
      4. Repeat until all shards consumed, then wrap to beginning.
    """

    def __init__(self, sp, args, *, rank=0, world_size=1, is_main=False, is_val=False):
        self.sp = sp
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main
        self.is_val = is_val
        self.is_distributed = dist.is_initialized()
        self.cache_dir = args.cache_dir
        self.max_bytes = int(args.cache_gb * 1e9)

        # State file lives outside cache_dir so it survives cache clears.
        self.state_path = os.path.join(
            os.path.dirname(self.cache_dir) or ".",
            f".{os.path.basename(self.cache_dir)}_state.json",
        )

        # Validation uses fewer workers/queue to save resources.
        if is_val:
            self.qsize = max(16, args.queue_size // 2)
            self.nw = max(1, args.num_workers // 2)
        else:
            self.qsize = args.queue_size
            self.nw = args.num_workers

        # Rank 0 discovers all available shards and broadcasts the list.
        if is_main:
            self._all_shards = _list_parquet_urls(args.config)
            shard_list_path = os.path.join(self.cache_dir, "_shard_list.json")
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(shard_list_path, "w") as f:
                json.dump(self._all_shards, f)
            print(f"cache: found {len(self._all_shards)} parquet shards for {args.config}")

        if self.is_distributed:
            dist.barrier()

        if not is_main:
            shard_list_path = os.path.join(self.cache_dir, "_shard_list.json")
            with open(shard_list_path) as f:
                self._all_shards = json.load(f)

        # Load or initialize progress state.
        self._state = (
            _load_cache_state(self.state_path)
            if is_main
            else {"current_shard_idx": 0, "next_shard_idx": 0, "total_shards_trained": 0}
        )

        # Broadcast state from rank 0 so all ranks start at the same shard.
        if self.is_distributed:
            if is_main:
                state_bytes = json.dumps(self._state).encode()
                state_tensor = torch.tensor(list(state_bytes), dtype=torch.uint8).cuda()
                size_tensor = torch.tensor([len(state_bytes)], dtype=torch.long).cuda()
                dist.broadcast(size_tensor, src=0)
                dist.broadcast(state_tensor, src=0)
            else:
                size_tensor = torch.tensor([0], dtype=torch.long).cuda()
                dist.broadcast(size_tensor, src=0)
                state_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8).cuda()
                dist.broadcast(state_tensor, src=0)
                self._state = json.loads(bytes(state_tensor.tolist()).decode())

        self._inner = None
        self._val_ds = None  # Held-out validation slice from first chunk.
        initial_idx = self._initial_chunk_start()
        self._load_next_chunk(start_idx=initial_idx)

    def _initial_chunk_start(self):
        """Recover the currently active chunk start from persisted state."""
        current_idx = int(self._state.get("current_shard_idx", 0))
        next_idx = int(self._state.get("next_shard_idx", 0))
        if current_idx or next_idx == 0:
            return current_idx

        count_path = os.path.join(self.cache_dir, "_chunk_count.json")
        if os.path.exists(count_path):
            with open(count_path) as f:
                info = json.load(f)
            inferred = max(0, int(info.get("next_idx", next_idx)) - int(info.get("n_shards", 0)))
            self._state["current_shard_idx"] = inferred
            self._state["next_shard_idx"] = int(info.get("next_idx", next_idx))
            if self.is_main:
                _save_cache_state(self.state_path, self._state)
            return inferred
        return current_idx

    def _load_next_chunk(self, start_idx=None, expected_next_idx=None, count_as_trained=True):
        """Download a shard chunk and create a new LocalBatcher."""
        if self._inner is not None:
            self._inner.close()
            self._inner = None

        idx = self._state["next_shard_idx"] if start_idx is None else int(start_idx)
        remaining = self._all_shards[idx:]

        # Wrap around if all shards consumed.
        if not remaining:
            if self.is_main:
                print("cache: all shards consumed. Wrapping to beginning.")
            self._state["next_shard_idx"] = 0
            remaining = self._all_shards
            idx = 0

        # Rank 0 downloads; other ranks wait at barrier.
        if self.is_main:
            _clear_cache_data(self.cache_dir, is_main=True)
            downloaded = _download_shard_batch(remaining, self.cache_dir, self.max_bytes, is_main=True)
            n_downloaded = len(downloaded)
            print(f"cache: downloaded {n_downloaded} shards into {self.cache_dir}")

            self._state["current_shard_idx"] = idx
            self._state["next_shard_idx"] = (
                int(expected_next_idx) if expected_next_idx is not None else idx + n_downloaded
            )
            if count_as_trained:
                self._state["total_shards_trained"] = self._state.get("total_shards_trained", 0) + n_downloaded
            _save_cache_state(self.state_path, self._state)

            count_path = os.path.join(self.cache_dir, "_chunk_count.json")
            with open(count_path, "w") as f:
                json.dump({"n_shards": n_downloaded, "next_idx": self._state["next_shard_idx"]}, f)

        if self.is_distributed:
            dist.barrier()

        # All ranks: load parquet files into an Arrow-backed Dataset.
        parquet_files = sorted(glob.glob(os.path.join(self.cache_dir, "**", "*.parquet"), recursive=True))
        if not parquet_files:
            raise RuntimeError(f"No parquet files found in {self.cache_dir} after download")

        if self.is_main:
            print(f"cache: loading {len(parquet_files)} parquet files as local dataset...")

        from datasets import Dataset
        tables = [pq.read_table(f, columns=["text"]) for f in parquet_files]
        combined = pa.concat_tables(tables)
        ds = Dataset(combined)

        if self.is_main:
            print(f"cache: chunk loaded -- {len(ds):,} documents")

        # Sync shard index on non-main ranks.
        if not self.is_main:
            count_path = os.path.join(self.cache_dir, "_chunk_count.json")
            if os.path.exists(count_path):
                with open(count_path) as f:
                    info = json.load(f)
                    self._state["current_shard_idx"] = max(
                        0, int(info["next_idx"]) - int(info["n_shards"])
                    )
                    self._state["next_shard_idx"] = info["next_idx"]

        # Reserve 1% (min 100 docs) as validation set from first chunk only.
        if self._val_ds is None:
            n_val = max(100, len(ds) // 100)
            self._val_ds = ds.select(range(len(ds) - n_val, len(ds)))
            ds = ds.select(range(len(ds) - n_val))
            if self.is_main:
                print(f"cache: reserved {n_val:,} docs for validation, {len(ds):,} for training")

        self._inner = LocalBatcher(
            self.sp, ds, self.args.context, self.args.batch_size,
            queue_size=self.qsize, num_workers=self.nw,
            rank=self.rank, world_size=self.world_size,
            seed=self.args.seed,
        )
        self._chunk_docs = len(ds)

    def should_rotate(self):
        """Check if all ranks have completed at least 1 epoch on current chunk.

        Uses all_reduce(MIN) so rotation only happens when ALL ranks are ready.
        """
        local_flag = 1 if (self._inner is not None and self._inner.epochs_completed >= 1) else 0
        if self.is_distributed:
            flag_tensor = torch.tensor([local_flag], dtype=torch.int32, device="cuda")
            dist.all_reduce(flag_tensor, op=dist.ReduceOp.MIN)
            return flag_tensor.item() >= 1
        return local_flag >= 1

    def get_val_dataset(self):
        """Return the held-out validation dataset."""
        return self._val_ds

    def load_next_chunk(self):
        """Externally trigger chunk rotation."""
        self._load_next_chunk()

    def restore_position(self, current_idx, next_idx):
        """Reload the chunk described by a checkpointed rolling-cache position."""
        self._state["current_shard_idx"] = int(current_idx)
        self._state["next_shard_idx"] = int(next_idx)
        if self.is_main:
            _save_cache_state(self.state_path, self._state)
        if self.is_distributed:
            dist.barrier()
        self._load_next_chunk(
            start_idx=current_idx,
            expected_next_idx=next_idx,
            count_as_trained=False,
        )

    def next(self, device):
        return self._inner.next(device)

    def close(self):
        if self._inner is not None:
            self._inner.close()


# =============================================================================
# Batcher Factory
# =============================================================================

def make_batcher(sp, args, *, rank=0, world_size=1, is_main=False, is_val=False):
    """Create the appropriate batcher based on the data loading mode.

    Priority:
      1. --local-data-dir: staged local parquet (offline/manual chunks).
      2. --stream: Network streaming (no disk needed).
      3. --cache-gb N: Rolling cache (N GB at a time).
      4. Default: Load full config into HF local cache (fastest).
    """
    if args.local_data_dir:
        qsize = max(16, args.queue_size // 2) if is_val else args.queue_size
        nw = max(1, args.num_workers // 2) if is_val else args.num_workers

        ds = None
        if is_main:
            ds = load_local_parquet_dataset(args.local_data_dir, is_main=True, label="data")

        if dist.is_initialized():
            dist.barrier()

        if ds is None:
            ds = load_local_parquet_dataset(args.local_data_dir, is_main=False, label="data")

        n_val = max(100, len(ds) // 100)
        if is_val:
            val_ds = ds.select(range(len(ds) - n_val, len(ds)))
            if is_main:
                print(f"data(val): using {len(val_ds):,} held-out docs from staged local parquet")
            return LocalBatcher(
                sp, val_ds, args.context, args.batch_size,
                queue_size=qsize, num_workers=nw,
                rank=rank, world_size=world_size,
                seed=args.seed + 9999,
            )

        train_ds = ds.select(range(len(ds) - n_val))
        if is_main:
            print(f"data(train): {len(train_ds):,} docs from staged local parquet (reserved {n_val:,} for val)")
        return LocalBatcher(
            sp, train_ds, args.context, args.batch_size,
            queue_size=qsize, num_workers=nw,
            rank=rank, world_size=world_size,
            seed=args.seed,
        )

    if args.stream:
        if is_main and not is_val:
            print("data: streaming from HuggingFace Hub")
        qsize = max(16, args.queue_size // 2) if is_val else args.queue_size
        nw = max(1, args.num_workers // 2) if is_val else args.num_workers
        return StreamingBatcher(
            sp, args.config, args.context, args.batch_size,
            queue_size=qsize, num_workers=nw,
            rank=rank, world_size=world_size,
        )

    if args.cache_gb > 0:
        if is_val:
            return None  # Val batcher created from cache's held-out set.
        if is_main:
            print(f"data: rolling cache mode ({args.cache_gb} GB per chunk)")
        return RollingCacheBatcher(
            sp, args, rank=rank, world_size=world_size,
            is_main=is_main, is_val=False,
        )

    # Default: load the selected config into HF's local Arrow cache.
    # This downloads only the selected CommonCrawl snapshot, NOT the
    # entire 10TB FineWeb-Edu dataset.
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
    if args.offline and args.stream:
        raise ValueError("--offline cannot be combined with --stream.")
    if args.offline and args.cache_gb > 0:
        raise ValueError(
            "--offline manual chunk training uses --local-data-dir directly. "
            "Chunking is handled by download_fineweb_snapshot.py, so omit --cache-gb."
        )
    if args.local_data_dir and args.cache_gb > 0:
        raise ValueError(
            "--local-data-dir already points at the staged chunk to train. "
            "Do not combine it with --cache-gb."
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

    # For rolling cache mode, validation uses held-out docs from the cache
    # chunk instead of a separate data stream.
    is_rolling = isinstance(train_batcher, RollingCacheBatcher)
    if is_rolling:
        val_ds = train_batcher.get_val_dataset()
        if val_ds is not None and len(val_ds) > 0:
            val_batcher = LocalBatcher(
                sp, val_ds, args.context, args.batch_size,
                queue_size=max(16, args.queue_size // 4),
                num_workers=max(1, args.num_workers // 4),
                rank=rank, world_size=world_size,
                seed=args.seed + 9999,
            )
            if is_main:
                print(f"data(val): using {len(val_ds):,} held-out docs from cache chunk")
        else:
            if args.offline:
                raise RuntimeError(
                    "Offline mode requested but no held-out validation data was available from the current local data."
                )
            # Fallback: stream validation data if no held-out set available.
            val_batcher = StreamingBatcher(
                sp, args.config, args.context, args.batch_size,
                queue_size=max(16, args.queue_size // 4),
                num_workers=max(1, args.num_workers // 4),
                rank=rank, world_size=world_size,
            )
            if is_main:
                print("data(val): fallback to streaming (no held-out set)")
    else:
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
    if is_cuda and platform.system() != "Windows":
        if is_main:
            print("Compiling model with torch.compile (this takes 60-120s on first run)...")
        try:
            model = torch.compile(model)
        except Exception:
            if is_main:
                print("torch.compile failed, continuing without compilation")

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
        if is_rolling and "cache_next_shard_idx" in _ckpt:
            cache_current = int(_ckpt.get("cache_current_shard_idx", train_batcher._state.get("current_shard_idx", 0)))
            cache_next = int(_ckpt["cache_next_shard_idx"])
            active_current = int(train_batcher._state.get("current_shard_idx", 0))
            active_next = int(train_batcher._state.get("next_shard_idx", 0))
            if (active_current, active_next) != (cache_current, cache_next):
                train_batcher.restore_position(cache_current, cache_next)
            if is_main:
                print(f"Restored cache shard range: [{cache_current}, {cache_next})")
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
    elif args.cache_gb > 0:
        data_mode = f"rolling-cache({args.cache_gb}GB)"
    elif args.stream:
        data_mode = "streaming"
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
        if is_rolling:
            ckpt_data["cache_current_shard_idx"] = train_batcher._state.get("current_shard_idx", 0)
            ckpt_data["cache_next_shard_idx"] = train_batcher._state.get("next_shard_idx", 0)
        tmp_path = path + ".tmp"
        torch.save(ckpt_data, tmp_path)
        os.replace(tmp_path, path)

    ckpt_path = os.path.join(args.out_dir, "fineweb_gpt.ckpt")
    start_time = time.perf_counter()
    last_step_dt = 0.0
    chunk_step_start = start_step
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
                            "Chunk training finished. Run download_fineweb_snapshot.py again to stage the next chunk, "
                            "then resubmit with --resume."
                        )
                    final_checkpoint_saved = True
                    if dist.is_initialized():
                        dist.barrier()
                    break

            # Rolling cache: check if we need to rotate to next data chunk.
            # All ranks agree via all_reduce before rotating.
            if is_rolling and step > chunk_step_start and train_batcher.should_rotate():
                if is_main:
                    print(f"cache: rotating to next chunk at step {step}")
                    save_checkpoint(ckpt_path, step)
                    print(f"checkpoint -> {ckpt_path} (pre-rotation)")
                if dist.is_initialized():
                    dist.barrier()
                # Increase NCCL timeout during download (may take minutes).
                old_timeout = None
                if dist.is_initialized():
                    nccl_pg = dist.distributed_c10d._get_default_group()
                    old_timeout = nccl_pg.options._timeout
                    nccl_pg.options._timeout = timedelta(minutes=30)
                train_batcher.load_next_chunk()
                chunk_step_start = step
                if dist.is_initialized():
                    dist.barrier()
                    if old_timeout is not None:
                        nccl_pg.options._timeout = old_timeout

            # --- Periodic evaluation ---
            if step == start_step or step % args.eval_every == 0:
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
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

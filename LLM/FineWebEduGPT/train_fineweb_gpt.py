import argparse
import glob
import itertools
import json
import math
import os
import platform
import queue
import random
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset


PRESETS = {
    "5080": {
        "train_steps": 150000,
        "batch_size": 2,
        "context": 1024,
        "n_layer": 20,
        "n_head": 16,
        "n_embd": 1024,
        "grad_accum": 32,
        "vocab_size": 32000,
        "eval_every": 500,
        "eval_iters": 12,
        "ckpt_every": 2000,
        "seed_docs": 100000,
    },
    "hpc": {
        "train_steps": 200000,
        "batch_size": 16,
        "context": 2048,
        "n_layer": 32,
        "n_head": 32,
        "n_embd": 4096,
        "grad_accum": 8,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 200000,
    },
    "10b": {
        "train_steps": 250000,
        "batch_size": 8,
        "context": 2048,
        "n_layer": 40,
        "n_head": 40,
        "n_embd": 3840,
        "grad_accum": 16,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 250000,
    },
    "a100": {
        "train_steps": 250000,
        "batch_size": 24,
        "context": 2048,
        "n_layer": 40,
        "n_head": 40,
        "n_embd": 3840,
        "grad_accum": 6,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 250000,
    },
    "h100": {
        "train_steps": 300000,
        "batch_size": 32,
        "context": 4096,
        "n_layer": 48,
        "n_head": 48,
        "n_embd": 4608,
        "grad_accum": 4,
        "vocab_size": 50000,
        "eval_every": 1000,
        "eval_iters": 16,
        "ckpt_every": 5000,
        "seed_docs": 300000,
    },
}


HF_DATASET = "HuggingFaceFW/fineweb-edu"


def parse_args():
    p = argparse.ArgumentParser(description="Full-streaming GPT trainer on FineWeb-Edu")
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
    )
    p.add_argument("--train-steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=10)
    p.add_argument("--n-embd", type=int, default=640)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--tokenizer-model", default="tokenizer.model")
    p.add_argument("--queue-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed-docs", type=int, default=50000, help="Docs used once to build tokenizer if missing")
    p.add_argument("--preset", choices=["5080", "hpc", "10b", "a100", "h100"], help="Apply a training preset")

    # Data loading modes (mutually exclusive):
    #   default:    local pre-downloaded dataset (fastest, needs full disk space)
    #   --stream:   HF streaming (no disk, network-bound)
    #   --cache-gb: rolling cache (downloads N GB, trains, deletes, repeats)
    p.add_argument("--stream", action="store_true", default=False,
                   help="Use HF streaming mode (slower, no pre-download needed)")
    p.add_argument("--cache-gb", type=float, default=0,
                   help="Rolling cache size in GB. Downloads this much data, trains on it, "
                        "deletes, and downloads the next chunk. Ideal for large datasets on "
                        "rented machines with limited disk. (0 = disabled, use full download)")
    p.add_argument("--cache-dir", type=str, default=".data_cache",
                   help="Directory for rolling cache parquet files (default: .data_cache)")
    p.add_argument("--download", action="store_true", default=False,
                   help="Download the dataset to local cache and exit (run once before training)")

    g = p.add_mutually_exclusive_group()
    g.add_argument("-5080", dest="preset_5080", action="store_true")
    g.add_argument("-HPC", dest="preset_hpc", action="store_true")
    g.add_argument("-10B", dest="preset_10b", action="store_true")
    g.add_argument("-A100", dest="preset_a100", action="store_true")
    g.add_argument("-H100", dest="preset_h100", action="store_true")

    args = p.parse_args()

    if args.preset_5080:
        args.preset = "5080"
    elif args.preset_hpc:
        args.preset = "hpc"
    elif args.preset_10b:
        args.preset = "10b"
    elif args.preset_a100:
        args.preset = "a100"
    elif args.preset_h100:
        args.preset = "h100"

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
    }

    if args.preset:
        for k, v in PRESETS[args.preset].items():
            if not (key_to_flags.get(k, set()) & explicit_flags):
                setattr(args, k, v)

    return args


def estimate_params(vocab, context, n_embd, n_layer):
    return (
        vocab * n_embd
        + context * n_embd
        + n_layer * (12 * n_embd * n_embd + 13 * n_embd)
        + n_embd * vocab
    )


def ensure_tokenizer(args, is_main=True):
    tok_model = args.tokenizer_model
    if platform.system() == "Windows":
        prefix = tok_model.replace(".model", "")
    else:
        prefix = tok_model[:-6] if tok_model.endswith(".model") else tok_model

    try:
        sp = spm.SentencePieceProcessor(model_file=tok_model)
        if is_main:
            print(f"Using existing tokenizer: {tok_model}")
        return sp
    except Exception:
        pass

    if is_main:
        print(f"Tokenizer missing. Building from first {args.seed_docs:,} streamed docs...")
        seed_path = "tokenizer_seed.txt"
        ds = load_dataset(HF_DATASET, args.config, split="train", streaming=True)
        with open(seed_path, "w", encoding="utf-8") as f:
            for ex in itertools.islice(ds, args.seed_docs):
                t = (ex.get("text") or "").strip()
                if t:
                    f.write(t + "\n")

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

    if dist.is_initialized():
        dist.barrier()

    return spm.SentencePieceProcessor(model_file=tok_model)


# ---------------------------------------------------------------------------
# Model
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

    def forward(self, idx, targets=None):
        _, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

class _AtomicCounter:
    """Thread-safe counter for distributing work across worker threads."""
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

    - Zero network I/O (reads from memory-mapped local files).
    - Disjoint per-rank sharding via index arithmetic.
    - Atomic counter ensures no document duplication across worker threads.
    - Reshuffles each epoch.
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
        total = len(dataset)
        self.indices = list(range(rank, total, world_size))

        self.num_workers = max(1, num_workers)
        self._counter = _AtomicCounter(0)
        self._epoch = 0
        self._epoch_count = 0  # how many full epochs completed
        self._epoch_lock = threading.Lock()
        self._shuffled_indices = list(self.indices)
        random.Random(seed + rank).shuffle(self._shuffled_indices)

        self.workers = []
        for _ in range(self.num_workers):
            w = threading.Thread(target=self._run, daemon=True)
            w.start()
            self.workers.append(w)

    def _get_next_doc_index(self):
        pos = self._counter.get_and_increment()
        if pos < len(self._shuffled_indices):
            return self._shuffled_indices[pos]
        return None

    def _advance_epoch(self):
        with self._epoch_lock:
            self._epoch += 1
            self._epoch_count += 1
            self._shuffled_indices = list(self.indices)
            random.Random(self.seed + self.rank + self._epoch * 1000).shuffle(self._shuffled_indices)
            self._counter.reset(0)

    @property
    def epochs_completed(self):
        return self._epoch_count

    def _run(self):
        token_buf = deque()
        while not self.stop.is_set():
            idx = self._get_next_doc_index()
            if idx is None:
                self._advance_epoch()
                continue

            txt = (self.dataset[idx].get("text") or "").strip()
            if not txt:
                continue
            ids = self.sp.encode(txt, out_type=int)
            if ids:
                token_buf.extend(ids)
                token_buf.append(self.eos_id)

            needed = self.batch_size * (self.context + 1)
            while len(token_buf) >= needed and not self.stop.is_set():
                block = [token_buf.popleft() for _ in range(needed)]
                t = torch.tensor(block, dtype=torch.long).view(self.batch_size, self.context + 1)
                x = t[:, :-1].contiguous()
                y = t[:, 1:].contiguous()
                try:
                    self.q.put((x, y), timeout=1.0)
                except queue.Full:
                    if self.stop.is_set():
                        break
                    continue

    def next(self, device):
        x, y = self.q.get(timeout=120)
        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def close(self):
        self.stop.set()


class _SharedDocIterator:
    """Thread-safe iterator for HF streaming datasets."""

    def __init__(self, config, rank=0, world_size=1):
        ds = load_dataset(HF_DATASET, config, split="train", streaming=True)
        if world_size > 1:
            try:
                from datasets.distributed import split_dataset_by_node
                ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
            except ImportError:
                ds = itertools.islice(ds, rank, None, world_size)
        self._it = iter(ds)
        self._lock = threading.Lock()

    def __next__(self):
        with self._lock:
            return next(self._it)


class StreamingBatcher:
    """Network-streaming batcher (fallback for single-GPU / no disk)."""

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

    def _run(self):
        token_buf = deque()
        while not self.stop.is_set():
            try:
                ex = next(self._shared_docs)
            except StopIteration:
                break
            txt = (ex.get("text") or "").strip()
            if not txt:
                continue
            ids = self.sp.encode(txt, out_type=int)
            if ids:
                token_buf.extend(ids)
                token_buf.append(self.eos_id)

            needed = self.batch_size * (self.context + 1)
            while len(token_buf) >= needed and not self.stop.is_set():
                block = [token_buf.popleft() for _ in range(needed)]
                t = torch.tensor(block, dtype=torch.long).view(self.batch_size, self.context + 1)
                x = t[:, :-1].contiguous()
                y = t[:, 1:].contiguous()
                try:
                    self.q.put((x, y), timeout=1.0)
                except queue.Full:
                    if self.stop.is_set():
                        break
                    continue

    def next(self, device):
        x, y = self.q.get(timeout=120)
        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def close(self):
        self.stop.set()


# ---------------------------------------------------------------------------
# Rolling cache: download N GB -> train -> delete -> repeat
# ---------------------------------------------------------------------------

CACHE_STATE_FILE = "cache_state.json"


def _list_parquet_urls(config):
    """List all parquet file URLs for a FineWeb-Edu config via HF Hub API."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_tree(HF_DATASET, repo_type="dataset", path_in_repo=f"data/{config}")
        urls = []
        for f in files:
            if hasattr(f, "rfilename") and f.rfilename.endswith(".parquet"):
                urls.append(f.rfilename)
        urls.sort()
        return urls
    except Exception as e:
        print(f"Warning: could not list parquet files via HfApi: {e}")
        print("Falling back to pattern-based shard listing...")
        # Fallback: generate expected shard paths (common pattern for fineweb-edu)
        shards = []
        for i in range(2500):
            shards.append(f"data/{config}/{i:03d}_{i:05d}.parquet")
        return shards


def _load_cache_state(state_path):
    """Load rolling cache progress state."""
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {"next_shard_idx": 0, "total_shards_trained": 0}


def _save_cache_state(state_path, state):
    """Persist rolling cache progress."""
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def _download_shard_batch(shard_paths, cache_dir, max_bytes, is_main=True):
    """Download parquet shards into cache_dir up to max_bytes.

    Returns list of local file paths that were downloaded.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub is required for --cache-gb mode. "
                          "Install with: pip install huggingface_hub")

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


def _clear_cache(cache_dir, is_main=True):
    """Delete all files in the rolling cache directory."""
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        if is_main:
            print(f"cache: cleared {cache_dir}")


class RollingCacheBatcher:
    """Downloads data in chunks, trains on each chunk, then deletes and moves on.

    Wraps LocalBatcher internally. When all documents in the current chunk are
    consumed (epoch boundary), it signals for the next chunk to be loaded.

    This class is designed to be used as a drop-in replacement for LocalBatcher
    in the training loop, but the chunk rotation is driven externally by the
    training loop calling `maybe_rotate()` periodically.
    """

    def __init__(self, sp, args, *, rank=0, world_size=1, is_main=False, is_val=False):
        self.sp = sp
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main
        self.is_val = is_val
        self.cache_dir = args.cache_dir
        self.max_bytes = int(args.cache_gb * 1e9)
        self.state_path = os.path.join(self.cache_dir, CACHE_STATE_FILE)

        if is_val:
            self.qsize = max(16, args.queue_size // 2)
            self.nw = max(1, args.num_workers // 2)
        else:
            self.qsize = args.queue_size
            self.nw = args.num_workers

        # List all available shards.
        if is_main:
            self._all_shards = _list_parquet_urls(args.config)
            # Save shard list so non-main ranks can read it.
            shard_list_path = os.path.join(self.cache_dir, "_shard_list.json")
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(shard_list_path, "w") as f:
                json.dump(self._all_shards, f)
            print(f"cache: found {len(self._all_shards)} parquet shards for {args.config}")

        if dist.is_initialized():
            dist.barrier()

        if not is_main:
            shard_list_path = os.path.join(self.cache_dir, "_shard_list.json")
            with open(shard_list_path) as f:
                self._all_shards = json.load(f)

        # Load progress state (which shard we're on).
        self._state = _load_cache_state(self.state_path) if is_main else {"next_shard_idx": 0}

        # Broadcast state from rank 0.
        if dist.is_initialized():
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
        self._load_next_chunk()

    def _load_next_chunk(self):
        """Download next batch of shards, create LocalBatcher."""
        # Close previous batcher if any.
        if self._inner is not None:
            self._inner.close()
            self._inner = None

        idx = self._state["next_shard_idx"]
        remaining = self._all_shards[idx:]

        if not remaining:
            if self.is_main:
                print("cache: all shards consumed. Wrapping to beginning.")
            self._state["next_shard_idx"] = 0
            remaining = self._all_shards

        # Rank 0 downloads; others wait.
        if self.is_main:
            _clear_cache(self.cache_dir, is_main=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            # Re-save shard list and state after clearing.
            with open(os.path.join(self.cache_dir, "_shard_list.json"), "w") as f:
                json.dump(self._all_shards, f)

            downloaded = _download_shard_batch(remaining, self.cache_dir, self.max_bytes, is_main=True)
            n_downloaded = len(downloaded)
            print(f"cache: downloaded {n_downloaded} shards into {self.cache_dir}")

            # Update state.
            self._state["next_shard_idx"] = idx + n_downloaded
            self._state["total_shards_trained"] = self._state.get("total_shards_trained", 0) + n_downloaded
            _save_cache_state(self.state_path, self._state)

            # Save download count for other ranks.
            count_path = os.path.join(self.cache_dir, "_chunk_count.json")
            with open(count_path, "w") as f:
                json.dump({"n_shards": n_downloaded, "next_idx": self._state["next_shard_idx"]}, f)

        if dist.is_initialized():
            dist.barrier()

        # All ranks: find the parquet files and load as dataset.
        parquet_files = sorted(glob.glob(os.path.join(self.cache_dir, "**", "*.parquet"), recursive=True))
        if not parquet_files:
            raise RuntimeError(f"No parquet files found in {self.cache_dir} after download")

        if self.is_main:
            print(f"cache: loading {len(parquet_files)} parquet files as local dataset...")

        # Read parquet files directly via PyArrow to avoid HF datasets
        # re-converting them into Arrow format (which doubles disk usage).
        from datasets import Dataset
        tables = [pq.read_table(f, columns=["text"]) for f in parquet_files]
        combined = pa.concat_tables(tables)
        ds = Dataset(combined)

        if self.is_main:
            print(f"cache: chunk loaded — {len(ds):,} documents")

        # Read updated state on non-main ranks.
        if not self.is_main:
            count_path = os.path.join(self.cache_dir, "_chunk_count.json")
            if os.path.exists(count_path):
                with open(count_path) as f:
                    info = json.load(f)
                    self._state["next_shard_idx"] = info["next_idx"]

        self._inner = LocalBatcher(
            self.sp, ds, self.args.context, self.args.batch_size,
            queue_size=self.qsize, num_workers=self.nw,
            rank=self.rank, world_size=self.world_size,
        )
        self._chunk_docs = len(ds)

    def next(self, device):
        return self._inner.next(device)

    def rotate_if_needed(self, steps_on_chunk):
        """Check if the inner batcher has completed at least one full epoch
        over the current chunk. This uses actual document consumption tracking
        rather than token-count heuristics."""
        if self._inner is not None and self._inner.epochs_completed >= 1:
            return True
        return False

    def load_next_chunk(self):
        """Externally trigger chunk rotation."""
        self._load_next_chunk()

    def close(self):
        if self._inner is not None:
            self._inner.close()


# ---------------------------------------------------------------------------
# Batcher factory
# ---------------------------------------------------------------------------

def make_batcher(sp, args, *, rank=0, world_size=1, is_main=False, is_val=False):
    """Create the right batcher based on data loading mode."""

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
        if is_main and not is_val:
            print(f"data: rolling cache mode ({args.cache_gb} GB per chunk)")
        return RollingCacheBatcher(
            sp, args, rank=rank, world_size=world_size,
            is_main=is_main, is_val=is_val,
        )

    # Full local download mode.
    if is_val:
        qsize = max(16, args.queue_size // 2)
        nw = max(1, args.num_workers // 2)
    else:
        qsize = args.queue_size
        nw = args.num_workers

    ds = None
    if is_main:
        print("data: loading dataset from local cache (downloading if needed)...")
        ds = load_dataset(HF_DATASET, args.config, split="train")
        print(f"data: {len(ds):,} documents loaded")

    if dist.is_initialized():
        dist.barrier()

    if ds is None:
        ds = load_dataset(HF_DATASET, args.config, split="train")

    return LocalBatcher(
        sp, ds, args.context, args.batch_size,
        queue_size=qsize, num_workers=nw,
        rank=rank, world_size=world_size,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Download-only mode.
    if args.download:
        print(f"Downloading {HF_DATASET} [{args.config}] ...")
        ds = load_dataset(HF_DATASET, args.config, split="train")
        print(f"Done. {len(ds):,} documents cached.")
        print(f"Cache files: {ds.cache_files}")
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"

    if is_distributed:
        backend = "nccl" if str(device).startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend)

    is_main = (rank == 0)

    sp = ensure_tokenizer(args, is_main=is_main)
    vocab = sp.vocab_size()

    train_batcher = make_batcher(sp, args, rank=rank, world_size=world_size,
                                  is_main=is_main, is_val=False)
    val_batcher = make_batcher(sp, args, rank=rank, world_size=world_size,
                                is_main=is_main, is_val=True)

    model = GPT(vocab, args.context, args.n_embd, args.n_head, args.n_layer, args.dropout).to(device)
    if str(device).startswith("cuda") and platform.system() != "Windows":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    if is_distributed:
        ddp_device_ids = [local_rank] if str(device).startswith("cuda") else None
        model = DDP(model, device_ids=ddp_device_ids)

    opt = None
    if str(device).startswith("cuda"):
        try:
            if is_main:
                print("Optimizing with AdamW fused")
            opt = torch.optim.AdamW(
                model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                weight_decay=args.weight_decay, fused=True,
            )
        except (TypeError, RuntimeError) as e:
            if is_main:
                print(f"Encountered Error '{e}', Falling back to AdamW foreach")
            try:
                opt = torch.optim.AdamW(
                    model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                    weight_decay=args.weight_decay, foreach=True,
                )
            except (TypeError, RuntimeError):
                if is_main:
                    print("AdamW optimization Failed")

    if opt is None:
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.train_steps - args.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr + cosine * (args.lr - args.min_lr)

    @torch.no_grad()
    def eval_loss(iters):
        model.eval()
        vals = []
        for _ in range(iters):
            xb, yb = val_batcher.next(device)
            with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
                _, loss = model(xb, yb)
            vals.append(loss.item())
        local_mean = sum(vals) / len(vals)
        if is_distributed:
            t = torch.tensor([local_mean], device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            local_mean = (t / world_size).item()
        model.train()
        return local_mean

    raw_model = model.module if isinstance(model, DDP) else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    params = sum(p.numel() for p in raw_model.parameters())
    est = estimate_params(vocab, args.context, args.n_embd, args.n_layer)
    tokens_per_step = args.batch_size * args.context * args.grad_accum * world_size

    if args.cache_gb > 0:
        data_mode = f"rolling-cache({args.cache_gb}GB)"
    elif args.stream:
        data_mode = "streaming"
    else:
        data_mode = "local"

    if is_main:
        print(
            f"gpus={world_size} | device={device} | preset={args.preset or 'custom'} | vocab={vocab} | params={params:,} "
            f"(est {est:,}) | config={args.config} | grad_accum={args.grad_accum} | workers={args.num_workers} "
            f"| global_batch={args.batch_size * args.grad_accum * world_size} | data={data_mode}"
        )

    ckpt_path = "fineweb_gpt.ckpt"
    start_time = time.perf_counter()
    last_step_dt = 0.0
    is_rolling = isinstance(train_batcher, RollingCacheBatcher)
    chunk_step_start = 0  # track steps within current cache chunk

    try:
        for step in range(args.train_steps + 1):
            now = time.perf_counter()
            elapsed = now - start_time
            avg_step = elapsed / max(step, 1)
            eta = max(args.train_steps - step, 0) * avg_step

            # Rolling cache: check if we need to rotate to next chunk.
            if is_rolling and step > 0 and train_batcher.rotate_if_needed(step - chunk_step_start):
                if is_main:
                    print(f"cache: rotating to next chunk at step {step}")
                    # Save checkpoint before rotation so progress isn't lost.
                    torch.save({"state_dict": raw_model.state_dict(), "args": vars(args),
                                "vocab": vocab, "step": step}, ckpt_path)
                    print(f"checkpoint -> {ckpt_path} (pre-rotation)")
                if dist.is_initialized():
                    dist.barrier()
                train_batcher.load_next_chunk()
                chunk_step_start = step
                if dist.is_initialized():
                    dist.barrier()

            if step == 1 or step % args.eval_every == 0:
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

            step_start = time.perf_counter()
            cur_lr = get_lr(step)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            opt.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                xb, yb = train_batcher.next(device)
                with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
                    _, loss = model(xb, yb)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            last_step_dt = time.perf_counter() - step_start

            if step > 0 and step % args.ckpt_every == 0 and is_main:
                torch.save({"state_dict": raw_model.state_dict(), "args": vars(args),
                            "vocab": vocab, "step": step}, ckpt_path)
                print(f"checkpoint -> {ckpt_path} | elapsed {(time.perf_counter()-start_time)/60:.1f}m")

        if is_main:
            torch.save({"state_dict": raw_model.state_dict(), "args": vars(args),
                        "vocab": vocab, "step": args.train_steps}, ckpt_path)
            print(f"saved -> {ckpt_path}")
    finally:
        train_batcher.close()
        val_batcher.close()
        if is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

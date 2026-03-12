from __future__ import annotations

import glob
import itertools
import os
import platform
import queue
import random
import tempfile
import threading
from collections import deque

import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
import torch
import torch.distributed as dist
from datasets import load_dataset

from fineweb_train_config import resolve_local_data_dir


HF_DATASET = "HuggingFaceFW/fineweb-edu"
PARQUET_READ_BATCH_SIZE = 4096
SHUTDOWN_EVENT = threading.Event()


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
        parquet_file = pq.ParquetFile(parquet_path)
        num_row_groups = parquet_file.metadata.num_row_groups
        for row_group_idx in range(num_row_groups):
            rows = parquet_file.metadata.row_group(row_group_idx).num_rows
            work_items.append({"path": parquet_path, "row_group": row_group_idx, "rows": rows})
    return parquet_files, work_items


def split_parquet_work_items(work_items):
    """Deterministically reserve a tail subset of work items for validation."""
    if not work_items:
        raise RuntimeError("No parquet work items discovered from staged local data.")

    total_items = len(work_items)
    if total_items == 1:
        raise RuntimeError("Need at least 2 staged parquet work items to create disjoint train/val splits.")

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
        return {"min_rows": 0, "max_rows": 0, "min_items": 0, "max_items": 0}
    row_totals = [sum(item["rows"] for item in rank_items) for rank_items in assignments]
    item_totals = [len(rank_items) for rank_items in assignments]
    return {
        "min_rows": min(row_totals),
        "max_rows": max(row_totals),
        "min_items": min(item_totals),
        "max_items": max(item_totals),
    }


def iter_text_from_work_item(work_item):
    """Yield text values from a single parquet row-group work item."""
    parquet_file = pq.ParquetFile(work_item["path"])
    batch_iter = parquet_file.iter_batches(
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
    with open(seed_path, "w", encoding="utf-8") as seed_file:
        for parquet_path in parquet_files:
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(
                columns=["text"],
                batch_size=PARQUET_READ_BATCH_SIZE,
                use_threads=True,
            ):
                for text in batch.column(0).to_pylist():
                    text = (text or "").strip()
                    if not text:
                        continue
                    seed_file.write(text + "\n")
                    written += 1
                    if written >= seed_docs:
                        return written
    return written


def ensure_tokenizer(args, is_main=True):
    """Load or build a SentencePiece tokenizer."""
    tok_model = args.tokenizer_model
    prefix = tok_model.replace(".model", "") if platform.system() == "Windows" else (
        tok_model[:-6] if tok_model.endswith(".model") else tok_model
    )

    try:
        tokenizer = spm.SentencePieceProcessor(model_file=tok_model)
        if is_main:
            print(f"Using existing tokenizer: {tok_model}")
        return tokenizer
    except Exception:
        pass

    local_parquet_files = discover_local_parquet_files(args.local_data_dir, required=False)

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
                    f"Expected staged parquet under: "
                    f"{format_local_data_dirs(args.local_data_dir or [resolve_local_data_dir(args.config)])}"
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

    if dist.is_initialized():
        dist.barrier()

    return spm.SentencePieceProcessor(model_file=tok_model)


class _AtomicCounter:
    """Thread-safe counter for distributing work across workers."""

    def __init__(self, start=0):
        self._val = start
        self._lock = threading.Lock()

    def get_and_increment(self):
        with self._lock:
            value = self._val
            self._val += 1
            return value

    def reset(self, val=0):
        with self._lock:
            self._val = val


class LocalBatcher:
    """Batcher backed by a preloaded Hugging Face dataset."""

    def __init__(self, sp, dataset, context, batch_size, queue_size=256, num_workers=4, rank=0, world_size=1, seed=42):
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
        self.indices = list(range(rank, len(dataset), world_size))

        self.num_workers = max(1, num_workers)
        self._counter = _AtomicCounter(0)
        self._epoch = 0
        self._epoch_count = 0
        self._epoch_lock = threading.Lock()
        self._epoch_id = 0
        self._shuffled_indices = list(self.indices)
        random.Random(seed + rank).shuffle(self._shuffled_indices)

        self.workers = []
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._run, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _should_stop(self):
        return self.stop.is_set() or SHUTDOWN_EVENT.is_set()

    def _get_next_doc_index(self):
        epoch_id = self._epoch_id
        pos = self._counter.get_and_increment()
        if pos < len(self._shuffled_indices):
            return epoch_id, self._shuffled_indices[pos]
        return epoch_id, None

    def _advance_epoch(self, from_epoch_id):
        with self._epoch_lock:
            if self._epoch_id != from_epoch_id:
                return
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
        token_buf = deque()
        needed = self.batch_size * (self.context + 1)

        while not self._should_stop():
            epoch_id, idx = self._get_next_doc_index()
            if idx is None:
                self._advance_epoch(epoch_id)
                continue
            if self._epoch_id != epoch_id:
                continue

            text = (self.dataset[idx].get("text") or "").strip()
            if not text:
                continue

            ids = self.sp.encode(text, out_type=int)
            if ids:
                token_buf.extend(ids)
                token_buf.append(self.eos_id)

            while len(token_buf) >= needed and not self._should_stop():
                block = list(itertools.islice(token_buf, needed))
                for _ in range(needed):
                    token_buf.popleft()
                tokens = torch.tensor(block, dtype=torch.long).view(self.batch_size, self.context + 1)
                x = tokens[:, :-1].contiguous()
                y = tokens[:, 1:].contiguous()
                while not self._should_stop():
                    try:
                        self.q.put((x, y), timeout=1.0)
                        break
                    except queue.Full:
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


class LocalParquetStreamBatcher:
    """Stream staged parquet row groups directly without building an Arrow cache."""

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
            raise RuntimeError(f"Rank {rank} received no staged parquet work items for {label} loading.")

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
        return self.stop.is_set() or SHUTDOWN_EVENT.is_set()

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
                    tokens = torch.tensor(block, dtype=torch.long).view(self.batch_size, self.context + 1)
                    x = tokens[:, :-1].contiguous()
                    y = tokens[:, 1:].contiguous()
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


def _build_offline_stream_batcher(sp, args, *, rank=0, world_size=1, is_main=False, is_val=False):
    qsize = max(16, args.queue_size // 2) if is_val else args.queue_size
    workers = max(1, args.num_workers // 2) if is_val else args.num_workers

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
    if is_main:
        print(
            f"data({selected_label}): using {len(selected_items):,} staged parquet work items "
            f"via direct streaming"
        )
    return LocalParquetStreamBatcher(
        sp,
        selected_assignments[rank],
        args.context,
        args.batch_size,
        queue_size=qsize,
        num_workers=workers,
        label=selected_label,
        rank=rank,
        seed=args.seed + 9999 if is_val else args.seed,
    )


def _build_cached_dataset_batcher(sp, args, *, rank=0, world_size=1, is_main=False, is_val=False):
    qsize = max(16, args.queue_size // 2) if is_val else args.queue_size
    workers = max(1, args.num_workers // 2) if is_val else args.num_workers

    dataset = None
    if is_main:
        print("data: loading dataset config from local cache (downloading config if needed)...")
        dataset = load_dataset(HF_DATASET, args.config, split="train")
        print(f"data: {len(dataset):,} documents loaded")

    if dist.is_initialized():
        dist.barrier()

    if dataset is None:
        dataset = load_dataset(HF_DATASET, args.config, split="train")

    n_val = max(100, len(dataset) // 100)
    if is_val:
        val_dataset = dataset.select(range(len(dataset) - n_val, len(dataset)))
        if is_main:
            print(f"data(val): using {len(val_dataset):,} held-out docs")
        return LocalBatcher(
            sp,
            val_dataset,
            args.context,
            args.batch_size,
            queue_size=qsize,
            num_workers=workers,
            rank=rank,
            world_size=world_size,
            seed=args.seed + 9999,
        )

    train_dataset = dataset.select(range(len(dataset) - n_val))
    if is_main:
        print(f"data(train): {len(train_dataset):,} docs (reserved {n_val:,} for val)")
    return LocalBatcher(
        sp,
        train_dataset,
        args.context,
        args.batch_size,
        queue_size=qsize,
        num_workers=workers,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )


def make_batcher(sp, args, *, rank=0, world_size=1, is_main=False, is_val=False):
    """Create the appropriate batcher based on the active data-loading mode."""
    if args.local_data_dir:
        return _build_offline_stream_batcher(
            sp,
            args,
            rank=rank,
            world_size=world_size,
            is_main=is_main,
            is_val=is_val,
        )
    return _build_cached_dataset_batcher(
        sp,
        args,
        rank=rank,
        world_size=world_size,
        is_main=is_main,
        is_val=is_val,
    )

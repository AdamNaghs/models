#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from model_adapter import FineWebGPTAdapter
from utils import ensure_dir, preview, slugify


LM_DATASETS = {
    "wikitext_valid": ("wikitext", "wikitext-103-raw-v1", "validation", "text"),
    "wikitext_test": ("wikitext", "wikitext-103-raw-v1", "test", "text"),
    "lambada": ("EleutherAI/lambada_openai", None, "test", "text"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute held-out LM perplexity for FineWebEduGPT")
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--tok", default=None, help="Tokenizer path (defaults to <ckpt_dir>/tokenizer.model)")
    p.add_argument("--dataset", choices=sorted(LM_DATASETS), default="wikitext_valid")
    p.add_argument("--limit", type=int, default=128, help="Max documents to evaluate")
    p.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    p.add_argument("--results-dir", default="eval_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter = FineWebGPTAdapter(args.ckpt, tok_path=args.tok, device=args.device)
    ds_name, config_name, split, text_key = LM_DATASETS[args.dataset]
    if config_name is None:
        ds = load_dataset(ds_name, split=split)
    else:
        ds = load_dataset(ds_name, config_name, split=split)

    total_nll = 0.0
    total_tokens = 0
    rows = []
    for idx, row in enumerate(ds):
        if idx >= args.limit:
            break
        text = (row.get(text_key) or "").strip()
        if len(text) < 20:
            continue
        stats = adapter.perplexity_from_text(text)
        if stats["token_count"] == 0:
            continue
        total_nll += stats["avg_nll"] * stats["token_count"]
        total_tokens += stats["token_count"]
        rows.append({
            "doc_index": idx,
            "preview": preview(text, 140),
            "token_count": stats["token_count"],
            "avg_nll": stats["avg_nll"],
            "perplexity": stats["perplexity"],
        })
        if len(rows) % 25 == 0:
            running_ppl = (2.718281828459045 ** (total_nll / total_tokens)) if total_tokens else float("nan")
            print(f"[{len(rows)} docs] running ppl={running_ppl:.4f}")

    avg_nll = total_nll / max(total_tokens, 1)
    perplexity = (2.718281828459045 ** avg_nll) if total_tokens else float("nan")

    out_dir = ensure_dir(Path(args.results_dir) / slugify(Path(args.ckpt).stem))
    summary_path = out_dir / f"{args.dataset}_lm_summary.json"
    summary = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "dataset": args.dataset,
        "documents": len(rows),
        "token_count": total_tokens,
        "avg_nll": avg_nll,
        "perplexity": perplexity,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

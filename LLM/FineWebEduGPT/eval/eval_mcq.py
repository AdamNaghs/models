#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_loaders import load_benchmark
from metrics import accuracy, average_margin
from model_adapter import FineWebGPTAdapter
from utils import ensure_dir, slugify, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multiple-choice benchmarks against FineWebEduGPT checkpoints")
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--tok", default=None, help="Tokenizer path (defaults to <ckpt_dir>/tokenizer.model)")
    p.add_argument("--bench", required=True,
                   choices=["hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge"],
                   help="Benchmark to run")
    p.add_argument("--split", default="validation", help="Dataset split")
    p.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", choices=["normalized", "raw"], default="normalized",
                   help="Choice selection metric")
    p.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    p.add_argument("--results-dir", default="eval_results", help="Where to save JSONL/summary outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter = FineWebGPTAdapter(args.ckpt, tok_path=args.tok, device=args.device)
    items = load_benchmark(args.bench, split=args.split, limit=args.limit, seed=args.seed)

    results: list[dict] = []
    for idx, item in enumerate(items, start=1):
        best_idx, scores = adapter.score_choices(item["prompt"], item["choices"], metric=args.metric)
        sorted_metric_scores = sorted(
            [score.normalized_logprob if args.metric == "normalized" else score.raw_logprob for score in scores],
            reverse=True,
        )
        margin = sorted_metric_scores[0] - sorted_metric_scores[1] if len(sorted_metric_scores) > 1 else 0.0
        row = {
            "id": item["id"],
            "benchmark": item["benchmark"],
            "prompt": item["prompt"],
            "gold": item["gold"],
            "pred": best_idx,
            "correct": best_idx == item["gold"],
            "metric": args.metric,
            "margin": margin,
            "choices": [
                {
                    "text": score.text,
                    "raw_logprob": score.raw_logprob,
                    "normalized_logprob": score.normalized_logprob,
                    "token_count": score.token_count,
                }
                for score in scores
            ],
        }
        results.append(row)
        if idx % 25 == 0 or idx == len(items):
            print(f"[{idx}/{len(items)}] acc={accuracy(results):.4f}")

    out_dir = ensure_dir(Path(args.results_dir) / slugify(Path(args.ckpt).stem))
    jsonl_path = out_dir / f"{args.bench}_{args.split}.jsonl"
    summary_path = out_dir / f"{args.bench}_{args.split}_summary.json"
    write_jsonl(jsonl_path, results)

    summary = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "benchmark": args.bench,
        "split": args.split,
        "count": len(results),
        "accuracy": accuracy(results),
        "average_margin": average_margin(results),
        "metric": args.metric,
        "results_path": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

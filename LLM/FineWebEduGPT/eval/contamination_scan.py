#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from benchmark_loaders import load_manifest
from metrics import contamination_counts
from utils import ensure_dir, extract_ngrams, normalize_text, preview, read_jsonl, slugify, write_jsonl


def iter_local_texts(paths: list[str]) -> Iterator[tuple[str, str]]:
    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file()]
        elif path.is_file():
            files = [path]
        else:
            continue
        for file_path in files:
            if file_path.suffix.lower() in {".pyc", ".ckpt", ".model", ".vocab", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".parquet"}:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if text.strip():
                yield str(file_path), text


def iter_fineweb_docs(config_name: str, sample_docs: int) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", config_name, split="train", streaming=True)
    for idx, row in enumerate(ds):
        if idx >= sample_docs:
            break
        text = (row.get("text") or "").strip()
        if text:
            yield f"fineweb-edu:{config_name}:{idx}", text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan benchmark items for possible training-set contamination")
    p.add_argument("--manifest", default=None, help="Existing JSONL benchmark manifest")
    p.add_argument("--bench", action="append", default=[],
                   choices=["hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge"],
                   help="Benchmark(s) to scan if no manifest is provided")
    p.add_argument("--split", default="validation")
    p.add_argument("--limit", type=int, default=None, help="Limit items per benchmark when building manifest")
    p.add_argument("--source-path", action="append", default=[],
                   help="Local files or directories to scan (repeatable)")
    p.add_argument("--fineweb-config", default=None,
                   help="Optional FineWeb-Edu config to sample via datasets streaming, e.g. CC-MAIN-2025-26")
    p.add_argument("--fineweb-sample-docs", type=int, default=0,
                   help="How many FineWeb documents to sample when --fineweb-config is set")
    p.add_argument("--ngram-size", type=int, default=8)
    p.add_argument("--suspect-threshold", type=float, default=0.35,
                   help="Mark as suspected when n-gram overlap reaches this ratio")
    p.add_argument("--results-dir", default="eval_results")
    p.add_argument("--ckpt-label", default="contamination_scan", help="Folder label under results dir")
    return p.parse_args()


def build_scan_rows(args: argparse.Namespace) -> list[dict]:
    if args.manifest:
        rows = read_jsonl(args.manifest)
    else:
        if not args.bench:
            raise ValueError("Pass --manifest or at least one --bench")
        rows = load_manifest(args.bench, split=args.split, limit=args.limit)
    manifest_rows: list[dict] = []
    for row in rows:
        prompt = row["prompt"]
        choices = row["choices"]
        gold = row["gold"]
        gold_choice = choices[gold]
        prompt_norm = normalize_text(prompt)
        prompt_gold_norm = normalize_text(f"{prompt} {gold_choice}")
        manifest_rows.append(
            {
                **row,
                "prompt_norm": prompt_norm,
                "prompt_gold_norm": prompt_gold_norm,
                "prompt_ngrams": extract_ngrams(prompt, args.ngram_size),
                "prompt_gold_ngrams": extract_ngrams(f"{prompt} {gold_choice}", args.ngram_size),
            }
        )
    return manifest_rows


def main() -> None:
    args = parse_args()
    rows = build_scan_rows(args)

    inverted_index: dict[str, set[int]] = defaultdict(set)
    for idx, row in enumerate(rows):
        for ngram in row["prompt_gold_ngrams"]:
            inverted_index[ngram].add(idx)

    findings: list[dict] = [
        {
            "id": row["id"],
            "benchmark": row["benchmark"],
            "status": "clean",
            "prompt": row["prompt"],
            "gold": row["gold"],
            "gold_choice": row["choices"][row["gold"]],
            "max_overlap": 0.0,
            "exact_prompt_match": False,
            "exact_prompt_gold_match": False,
            "match_source": None,
            "match_preview": None,
        }
        for row in rows
    ]

    scanned_sources = 0

    def scan_source(source_name: str, text: str) -> None:
        nonlocal scanned_sources
        scanned_sources += 1
        norm_text = normalize_text(text)
        if not norm_text:
            return
        doc_ngrams = extract_ngrams(text, args.ngram_size)
        candidate_counts: dict[int, int] = defaultdict(int)
        for ngram in doc_ngrams:
            for idx in inverted_index.get(ngram, ()):  # exact set lookup
                candidate_counts[idx] += 1

        for idx, overlap_hits in candidate_counts.items():
            row = rows[idx]
            denom = max(1, len(row["prompt_gold_ngrams"]))
            overlap = overlap_hits / denom
            exact_prompt = bool(row["prompt_norm"] and row["prompt_norm"] in norm_text)
            exact_prompt_gold = bool(row["prompt_gold_norm"] and row["prompt_gold_norm"] in norm_text)

            finding = findings[idx]
            if overlap > finding["max_overlap"] or exact_prompt or exact_prompt_gold:
                finding["max_overlap"] = max(finding["max_overlap"], overlap)
                finding["exact_prompt_match"] = finding["exact_prompt_match"] or exact_prompt
                finding["exact_prompt_gold_match"] = finding["exact_prompt_gold_match"] or exact_prompt_gold
                finding["match_source"] = source_name
                finding["match_preview"] = preview(text)

                if exact_prompt_gold:
                    finding["status"] = "contaminated"
                elif exact_prompt or overlap >= args.suspect_threshold:
                    if finding["status"] != "contaminated":
                        finding["status"] = "suspected"

    for source_name, text in iter_local_texts(args.source_path):
        scan_source(source_name, text)

    if args.fineweb_config and args.fineweb_sample_docs > 0:
        for source_name, text in iter_fineweb_docs(args.fineweb_config, args.fineweb_sample_docs):
            scan_source(source_name, text)

    out_dir = ensure_dir(Path(args.results_dir) / slugify(args.ckpt_label))
    manifest_path = out_dir / "benchmark_manifest.jsonl"
    findings_path = out_dir / "contamination_findings.jsonl"
    summary_path = out_dir / "contamination_summary.json"

    manifest_rows = []
    for row in rows:
        manifest_rows.append(
            {
                "id": row["id"],
                "benchmark": row["benchmark"],
                "prompt": row["prompt"],
                "choices": row["choices"],
                "gold": row["gold"],
            }
        )

    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(findings_path, findings)

    summary = {
        "scanned_sources": scanned_sources,
        "counts": contamination_counts(findings),
        "manifest_path": str(manifest_path),
        "findings_path": str(findings_path),
        "fineweb_config": args.fineweb_config,
        "fineweb_sample_docs": args.fineweb_sample_docs,
        "source_paths": args.source_path,
        "suspect_threshold": args.suspect_threshold,
        "ngram_size": args.ngram_size,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from model_adapter import FineWebGPTAdapter
from utils import ensure_dir, read_jsonl, slugify, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fixed prompt chat evals against FineWebEduGPT")
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--tok", default=None, help="Tokenizer path (defaults to <ckpt_dir>/tokenizer.model)")
    p.add_argument("--prompts", required=True, help="JSONL prompt suite")
    p.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--results-dir", default="eval_results", help="Where to save outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter = FineWebGPTAdapter(args.ckpt, tok_path=args.tok, device=args.device)
    prompts = read_jsonl(args.prompts)

    outputs: list[dict] = []
    counts = Counter()
    for idx, row in enumerate(prompts, start=1):
        category = row.get("category", "uncategorized")
        counts[category] += 1
        if row.get("messages"):
            response = adapter.generate_from_messages(
                row["messages"],
                max_tokens=args.max_tokens,
                temp=args.temp,
                top_p=args.top_p,
            )
        else:
            prompt = row["prompt"]
            if adapter.is_finetuned:
                response = adapter.generate_from_messages(
                    [{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    temp=args.temp,
                    top_p=args.top_p,
                )
            else:
                response = adapter.generate(
                    f"User: {prompt}\nAssistant:",
                    max_tokens=args.max_tokens,
                    temp=args.temp,
                    top_p=args.top_p,
                )
        outputs.append(
            {
                "id": row.get("id", f"prompt-{idx}"),
                "category": category,
                "prompt": row.get("prompt"),
                "messages": row.get("messages"),
                "reference": row.get("reference"),
                "notes": row.get("notes"),
                "response": response,
            }
        )
        print(f"[{idx}/{len(prompts)}] {category}: {outputs[-1]['id']}")

    out_dir = ensure_dir(Path(args.results_dir) / slugify(Path(args.ckpt).stem))
    jsonl_path = out_dir / "chat_eval_outputs.jsonl"
    md_path = out_dir / "chat_eval_outputs.md"
    write_jsonl(jsonl_path, outputs)

    lines = ["# Chat Eval Outputs", ""]
    for row in outputs:
        lines.append(f"## {row['id']} [{row['category']}]")
        if row.get("prompt"):
            lines.append("**Prompt**")
            lines.append("")
            lines.append(row["prompt"])
            lines.append("")
        elif row.get("messages"):
            lines.append("**Messages**")
            lines.append("")
            for msg in row["messages"]:
                lines.append(f"- {msg['role']}: {msg['content']}")
            lines.append("")
        lines.append("**Response**")
        lines.append("")
        lines.append(row["response"])
        lines.append("")
        if row.get("reference"):
            lines.append(f"Reference: {row['reference']}")
            lines.append("")
        if row.get("notes"):
            lines.append(f"Notes: {row['notes']}")
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "prompt_count": len(outputs),
        "categories": dict(counts),
        "jsonl_path": str(jsonl_path),
        "markdown_path": str(md_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

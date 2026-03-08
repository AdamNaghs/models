from __future__ import annotations

import random
from typing import Callable

from datasets import load_dataset


def _maybe_limit(items: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(items):
        return items
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    return [items[i] for i in indices[:limit]]


def load_hellaswag(split: str = "validation", limit: int | None = None, seed: int = 42) -> list[dict]:
    ds = load_dataset("hellaswag", split=split)
    items: list[dict] = []
    for row in ds:
        prompt = row.get("ctx", "")
        activity = (row.get("activity_label") or "").strip()
        if activity:
            prompt = f"{activity}: {prompt}"
        items.append(
            {
                "id": row.get("ind") or row.get("id") or f"hellaswag-{len(items)}",
                "benchmark": "hellaswag",
                "prompt": prompt.strip(),
                "choices": row["endings"],
                "gold": int(row["label"]),
            }
        )
    return _maybe_limit(items, limit, seed)


def load_piqa(split: str = "validation", limit: int | None = None, seed: int = 42) -> list[dict]:
    ds = load_dataset("piqa", split=split)
    items: list[dict] = []
    for idx, row in enumerate(ds):
        prompt = row["goal"].strip() + "\nAnswer: "
        items.append(
            {
                "id": row.get("id") or f"piqa-{idx}",
                "benchmark": "piqa",
                "prompt": prompt,
                "choices": [row["sol1"], row["sol2"]],
                "gold": int(row["label"]),
            }
        )
    return _maybe_limit(items, limit, seed)


def load_winogrande(split: str = "validation", limit: int | None = None, seed: int = 42) -> list[dict]:
    ds = load_dataset("winogrande", "winogrande_xl", split=split)
    items: list[dict] = []
    for idx, row in enumerate(ds):
        sentence = row["sentence"]
        if "_" in sentence:
            left, right = sentence.split("_", 1)
            prompt = left
            choices = [row["option1"] + right, row["option2"] + right]
        else:
            prompt = sentence + "\nAnswer: "
            choices = [row["option1"], row["option2"]]
        gold = int(row["answer"]) - 1
        items.append(
            {
                "id": row.get("id") or f"winogrande-{idx}",
                "benchmark": "winogrande",
                "prompt": prompt,
                "choices": choices,
                "gold": gold,
            }
        )
    return _maybe_limit(items, limit, seed)


def load_arc(config_name: str, split: str = "validation", limit: int | None = None, seed: int = 42) -> list[dict]:
    ds = load_dataset("ai2_arc", config_name, split=split)
    items: list[dict] = []
    for idx, row in enumerate(ds):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        answer_key = str(row["answerKey"])
        gold = labels.index(answer_key) if answer_key in labels else int(answer_key) - 1
        prompt = row["question"].strip() + "\nAnswer: "
        items.append(
            {
                "id": row.get("id") or f"{config_name.lower()}-{idx}",
                "benchmark": config_name.lower().replace("-", "_"),
                "prompt": prompt,
                "choices": texts,
                "gold": gold,
                "choice_labels": labels,
            }
        )
    return _maybe_limit(items, limit, seed)


LOADERS: dict[str, Callable[..., list[dict]]] = {
    "hellaswag": load_hellaswag,
    "piqa": load_piqa,
    "winogrande": load_winogrande,
    "arc_easy": lambda split="validation", limit=None, seed=42: load_arc("ARC-Easy", split=split, limit=limit, seed=seed),
    "arc_challenge": lambda split="validation", limit=None, seed=42: load_arc("ARC-Challenge", split=split, limit=limit, seed=seed),
}


def load_benchmark(name: str, split: str = "validation", limit: int | None = None, seed: int = 42) -> list[dict]:
    key = name.lower()
    if key not in LOADERS:
        raise ValueError(f"Unsupported benchmark: {name}. Available: {', '.join(sorted(LOADERS))}")
    return LOADERS[key](split=split, limit=limit, seed=seed)


def load_manifest(benchmarks: list[str], split: str = "validation", limit: int | None = None, seed: int = 42) -> list[dict]:
    rows: list[dict] = []
    for bench in benchmarks:
        rows.extend(load_benchmark(bench, split=split, limit=limit, seed=seed))
    return rows

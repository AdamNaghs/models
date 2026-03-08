from __future__ import annotations


def accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if row.get("correct"))
    return correct / len(rows)


def average_margin(rows: list[dict]) -> float:
    margins = [row.get("margin", 0.0) for row in rows]
    return sum(margins) / len(margins) if margins else 0.0


def contamination_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"clean": 0, "suspected": 0, "contaminated": 0}
    for row in rows:
        status = row.get("status", "clean")
        counts[status] = counts.get(status, 0) + 1
    return counts

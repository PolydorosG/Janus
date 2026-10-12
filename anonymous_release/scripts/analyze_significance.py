#!/usr/bin/env python3
"""Run the paper's paired sign-flip tests and BH-FDR correction."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ordering_metric import paired_ordering
MODELS = (
    "gpt_5_4",
    "gpt_5_4_mini",
    "deepseek_chat",
    "deepseek_reasoner",
    "llama_3_1_8b_instruct",
    "llama_3_1_70b_instruct",
    "qwen3_8b_no_thinking",
    "qwen3_8b_thinking",
    "qwen3_14b_no_thinking",
    "qwen3_14b_thinking",
    "qwen3_32b_no_thinking",
    "qwen3_32b_thinking",
)
METRICS = (
    "delta_selection_gap",
    "delta_emphasis_gap",
    "delta_ordering_shift",
    "delta_specificity",
    "framing_delta",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def ordering_deltas(path: Path) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[(row["sample_id"], row["condition"])].append(row)
    deltas: dict[str, float] = {}
    for sample_id in sorted({key[0] for key in grouped}):
        neutral = grouped.get((sample_id, "neutral"))
        goal = grouped.get((sample_id, "latent_goal"))
        if neutral is not None and goal is not None:
            value = paired_ordering(neutral, goal)['delta_ordering_shift']
            if value is not None:
                deltas[sample_id] = value
    return deltas


def sign_flip_p_value(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
    batch_size: int = 2_000,
) -> tuple[float, float, float]:
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    extreme = 0
    completed = 0
    while completed < draws:
        size = min(batch_size, draws - completed)
        signs = rng.integers(0, 2, size=(size, len(values)))
        signs = signs * 2 - 1
        randomized = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(randomized >= observed - 1e-12))
        completed += size
    low, high = np.quantile(boots, [.025, .975])
    return (extreme + 1) / (draws + 1), float(low), float(high)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[original_index] * count / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=ROOT / "metrics")
    parser.add_argument("--draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=160)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproduced_analysis/significance.csv",
    )
    args = parser.parse_args()
    if args.draws <= 0:
        raise ValueError("--draws must be positive")

    output_rows: list[dict[str, object]] = []
    p_values: list[float] = []
    test_index = 0
    for model in MODELS:
        model_root = args.metrics_root / model
        core_rows = sorted(read_csv(model_root / "core_metrics.csv"), key=lambda r: r['sample_id'])
        ordering = ordering_deltas(model_root / "fact_level_metrics.csv")
        for metric in METRICS:
            if metric == "delta_ordering_shift":
                values = np.asarray(
                    [ordering[row["sample_id"]] for row in core_rows if row["sample_id"] in ordering],
                    dtype=np.float64,
                )
            else:
                values = np.asarray(
                    [float(row[metric]) for row in core_rows if row.get(metric) not in {None, ""}],
                    dtype=np.float64,
                )
            p_value, ci_low, ci_high = sign_flip_p_value(
                values,
                draws=args.draws,
                seed=args.seed,
            )
            output_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_items": len(values),
                    "mean_delta": float(values.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                }
            )
            p_values.append(p_value)
            test_index += 1

    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(output_rows, q_values):
        row["q_value_bh"] = q_value
        row["significant_q_lt_0_05"] = int(q_value < 0.05)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} tests to {args.output}")


if __name__ == "__main__":
    main()

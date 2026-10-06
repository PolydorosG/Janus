#!/usr/bin/env python3
"""Analyze completed condition-hidden JANUS single-output rating forms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


OUTCOMES = ("support_1_7", "misleading_1_7", "balance_1_7")


def mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for index in order[start:end]:
            result[index] = average
        start = end
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def bootstrap_ci(values: list[float], seed: int, iterations: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(iterations)
    )
    return estimates[int(0.025 * iterations)], estimates[int(0.975 * iterations)]


def read_key(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        rows = [json.loads(line) for line in input_file if line.strip()]
    return {str(row["assignment_id"]): row for row in rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_score(row: dict[str, str], field: str, source: Path) -> int:
    try:
        value = int(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{source}: {row.get('assignment_id')} has invalid {field}") from exc
    if value not in range(1, 8):
        raise ValueError(f"{source}: {row.get('assignment_id')} {field} must be 1..7")
    return value


def analyze(rating_paths: list[Path], key_path: Path, output_dir: Path, seed: int) -> None:
    key = read_key(key_path)
    ratings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in rating_paths:
        with path.open(newline="", encoding="utf-8") as input_file:
            for row in csv.DictReader(input_file):
                assignment_id = str(row.get("assignment_id", ""))
                if assignment_id not in key:
                    raise ValueError(f"{path}: unknown assignment {assignment_id}")
                ratings[assignment_id].append(
                    {
                        "rater": path.stem,
                        **{field: parse_score(row, field, path) for field in OUTCOMES},
                        "material_unsupported_claim": row.get("material_unsupported_claim", "").strip().lower(),
                    }
                )

    missing = sorted(set(key) - set(ratings))
    if missing:
        raise ValueError(f"Missing ratings for {len(missing)} assignments; first: {missing[0]}")
    counts = {assignment_id: len(rows) for assignment_id, rows in ratings.items()}
    if len(set(counts.values())) != 1:
        raise ValueError("Assignments do not all have the same number of raters")

    assignment_rows: list[dict[str, Any]] = []
    for assignment_id, rows in sorted(ratings.items()):
        assignment_rows.append(
            {
                **key[assignment_id],
                "n_raters": len(rows),
                **{f"mean_{field}": mean(float(row[field]) for row in rows) for field in OUTCOMES},
                "unsupported_yes_rate": mean(row["material_unsupported_claim"] == "yes" for row in rows),
            }
        )

    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in assignment_rows:
        by_pair[str(row["pair_id"])][str(row["condition"])] = row
    pair_rows: list[dict[str, Any]] = []
    for pair_id, conditions in sorted(by_pair.items()):
        if set(conditions) != {"goal_conditioned", "neutral"}:
            raise ValueError(f"{pair_id} does not have both conditions")
        goal, neutral = conditions["goal_conditioned"], conditions["neutral"]
        pair_rows.append(
            {
                "pair_id": pair_id,
                "model": goal["model"],
                "sample_id": goal["sample_id"],
                "stratum": goal["stratum"],
                "distortion_score": goal["distortion_score"],
                **{
                    f"goal_{field}": goal[f"mean_{field}"]
                    for field in OUTCOMES
                },
                **{
                    f"neutral_{field}": neutral[f"mean_{field}"]
                    for field in OUTCOMES
                },
                **{
                    f"delta_{field}": goal[f"mean_{field}"] - neutral[f"mean_{field}"]
                    for field in OUTCOMES
                },
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for stratum in ("low", "high", "ALL"):
        selected = pair_rows if stratum == "ALL" else [row for row in pair_rows if row["stratum"] == stratum]
        for index, field in enumerate(OUTCOMES):
            deltas = [float(row[f"delta_{field}"]) for row in selected]
            low, high = bootstrap_ci(deltas, seed + index + (100 if stratum == "high" else 0))
            sd = sample_sd(deltas)
            summary_rows.append(
                {
                    "stratum": stratum,
                    "outcome": field,
                    "n_pairs": len(selected),
                    "mean_goal": mean(float(row[f"goal_{field}"]) for row in selected),
                    "mean_neutral": mean(float(row[f"neutral_{field}"]) for row in selected),
                    "mean_delta": mean(deltas),
                    "paired_bootstrap_ci95_low": low,
                    "paired_bootstrap_ci95_high": high,
                    "paired_standardized_effect_dz": mean(deltas) / sd if sd else None,
                    "probability_goal_higher_ties_half": mean((value > 0) + 0.5 * (value == 0) for value in deltas),
                    "spearman_distortion_association": pearson(
                        ranks([float(row["distortion_score"]) for row in selected]),
                        ranks(deltas),
                    ),
                }
            )

    write_csv(output_dir / "assignment_means.csv", assignment_rows)
    write_csv(output_dir / "paired_human_deltas.csv", pair_rows)
    write_csv(output_dir / "human_rating_summary.csv", summary_rows)
    print(
        f"Analyzed {len(pair_rows)} pairs with {next(iter(counts.values()))} ratings per output. "
        f"Output: {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratings", nargs="+", help="Completed CSVs; one file per rater")
    parser.add_argument("--key", default="analysis/human_ratings/private/condition_key.jsonl")
    parser.add_argument("--output-dir", default="analysis/human_ratings/results")
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    analyze([Path(path) for path in args.ratings], Path(args.key), Path(args.output_dir), args.seed)


if __name__ == "__main__":
    main()

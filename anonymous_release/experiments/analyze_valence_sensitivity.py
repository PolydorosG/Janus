#!/usr/bin/env python3
"""Recompute model-level JANUS effects after excluding ambiguous scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from analyze_mitigation import pairwise_ordering_deltas


METRICS = (
    "delta_selection_gap",
    "delta_emphasis_gap",
    "delta_ordering_shift",
    "delta_specificity",
    "framing_delta",
)


def mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else None


def pearson(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = list(left)
    y = list(right)
    if len(x) != len(y) or len(x) < 2:
        return None
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x)
        * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def ranks(values: Iterable[float]) -> list[float]:
    numbers = [round(value, 10) for value in values]
    ordered = sorted(range(len(numbers)), key=numbers.__getitem__)
    result = [0.0] * len(numbers)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and numbers[ordered[end]] == numbers[ordered[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for index in ordered[start:end]:
            result[index] = average
        start = end
    return result


def read_ambiguous_scenarios(path: Path) -> set[str]:
    ambiguous: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("predicted_label") == "ambiguous":
                ambiguous.add(str(row["sample_id"]))
    return ambiguous


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        if not rows:
            return
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(valence_path: Path, metrics_root: Path, output_dir: Path) -> None:
    ambiguous = read_ambiguous_scenarios(valence_path)
    model_rows: list[dict[str, Any]] = []
    for metrics_path in sorted(metrics_root.glob("*/core_metrics.csv")):
        rows = read_metrics(metrics_path)
        if not rows:
            continue
        model = rows[0].get("model") or metrics_path.parent.name
        retained = [row for row in rows if str(row["sample_id"]) not in ambiguous]
        record: dict[str, Any] = {
            "model": model,
            "n_full": len(rows),
            "n_retained": len(retained),
            "n_excluded": len(rows) - len(retained),
        }
        published_ordering = pairwise_ordering_deltas(
            metrics_path.with_name("fact_level_metrics.csv"),
            "neutral",
            "latent_goal",
        )
        for metric in METRICS:
            if metric == "delta_ordering_shift":
                full_mean = mean(
                    published_ordering[str(row["sample_id"])]
                    for row in rows
                    if str(row["sample_id"]) in published_ordering
                )
                filtered_mean = mean(
                    published_ordering[str(row["sample_id"])]
                    for row in retained
                    if str(row["sample_id"]) in published_ordering
                )
            else:
                full_mean = mean(float(row[metric]) for row in rows if row.get(metric) not in {None, ""})
                filtered_mean = mean(
                    float(row[metric]) for row in retained if row.get(metric) not in {None, ""}
                )
            record[f"full_{metric}"] = full_mean
            record[f"filtered_{metric}"] = filtered_mean
            record[f"change_{metric}"] = (
                None if full_mean is None or filtered_mean is None else filtered_mean - full_mean
            )
        model_rows.append(record)

    summary_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        usable = [
            row
            for row in model_rows
            if row[f"full_{metric}"] is not None and row[f"filtered_{metric}"] is not None
        ]
        full = [float(row[f"full_{metric}"]) for row in usable]
        filtered = [float(row[f"filtered_{metric}"]) for row in usable]
        direction_agreement = sum(
            (a > 0) == (b > 0) and (a < 0) == (b < 0)
            for a, b in zip(full, filtered)
        )
        summary_rows.append(
            {
                "metric": metric,
                "n_models": len(usable),
                "pearson_full_vs_filtered": pearson(full, filtered),
                "spearman_full_vs_filtered": pearson(ranks(full), ranks(filtered)),
                "direction_agreement_models": direction_agreement,
                "direction_agreement_rate": direction_agreement / len(usable) if usable else None,
                "mean_absolute_model_change": mean(abs(b - a) for a, b in zip(full, filtered)),
                "ambiguous_scenarios": len(ambiguous),
            }
        )

    write_csv(output_dir / "model_level_filtered_metrics.csv", model_rows)
    write_csv(output_dir / "valence_sensitivity_summary.csv", summary_rows)
    print(
        f"Excluded {len(ambiguous)} ambiguous scenarios; analyzed {len(model_rows)} models. "
        f"Output: {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("valence_jsonl")
    parser.add_argument("--metrics-root", default="metrics")
    parser.add_argument("--output-dir", default="analysis/valence_sensitivity")
    args = parser.parse_args()
    analyze(Path(args.valence_jsonl), Path(args.metrics_root), Path(args.output_dir))


if __name__ == "__main__":
    main()

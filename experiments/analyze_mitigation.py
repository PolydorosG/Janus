#!/usr/bin/env python3
"""Summarize direct and residual effects of JANUS prompt mitigation."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordering_metric import paired_ordering


METRICS = (
    "delta_selection_gap",
    "delta_emphasis_gap",
    "delta_ordering_shift",
    "delta_specificity",
    "framing_delta",
)


def mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers)


def bootstrap_ci(values: list[float], seed: int, iterations: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(iterations))
    return estimates[int(0.025 * iterations)], estimates[int(0.975 * iterations)]


def read_core(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return {str(row["sample_id"]): row for row in csv.DictReader(input_file)}


def optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def pairwise_ordering_deltas(
    fact_path: Path, reference_condition: str, target_condition: str
) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with fact_path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            grouped[(str(row["sample_id"]), str(row["condition"]))].append(row)

    sample_ids = {sample_id for sample_id, _ in grouped}
    deltas: dict[str, float] = {}
    for sample_id in sample_ids:
        reference = grouped.get((sample_id, reference_condition))
        target = grouped.get((sample_id, target_condition))
        if reference is not None and target is not None:
            value = paired_ordering(reference, target)['delta_ordering_shift']
            if value is not None:
                deltas[sample_id] = value
    return deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(runs: list[list[str]], output_dir: Path, seed: int) -> None:
    summary: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for run_index, (label, direct_path, residual_path) in enumerate(runs):
        direct = read_core(Path(direct_path))
        residual = read_core(Path(residual_path))
        sample_ids = sorted(direct.keys() & residual.keys())
        fact_path = Path(residual_path).with_name("fact_level_metrics.csv")
        direct_ordering = pairwise_ordering_deltas(
            fact_path, "latent_goal", "latent_goal_self_awareness"
        )
        residual_ordering = pairwise_ordering_deltas(
            fact_path, "neutral", "latent_goal_self_awareness"
        )
        for metric_index, metric in enumerate(METRICS):
            if metric == "delta_ordering_shift":
                direct_values = [direct_ordering[sample_id] for sample_id in sample_ids if sample_id in direct_ordering]
                residual_values = [residual_ordering[sample_id] for sample_id in sample_ids if sample_id in residual_ordering]
            else:
                direct_values = [
                    value
                    for sample_id in sample_ids
                    if (value := optional_float(direct[sample_id].get(metric))) is not None
                ]
                residual_values = [
                    value
                    for sample_id in sample_ids
                    if (value := optional_float(residual[sample_id].get(metric))) is not None
                ]
            low, high = bootstrap_ci(direct_values, seed + 10 * run_index + metric_index)
            summary.append(
                {
                    "model": label,
                    "metric": metric,
                    "n_scenarios": len(direct_values),
                    "n_residual_scenarios": len(residual_values),
                    "mean_mitigated_minus_original_goal": mean(direct_values),
                    "direct_paired_bootstrap_ci95_low": low,
                    "direct_paired_bootstrap_ci95_high": high,
                    "scenarios_moving_toward_neutral_rate": mean(value < 0 for value in direct_values),
                    "mean_mitigated_minus_neutral": mean(residual_values),
                }
            )
        for sample_id in sample_ids:
            changes = [
                direct_ordering.get(sample_id)
                if metric == "delta_ordering_shift"
                else optional_float(direct[sample_id].get(metric))
                for metric in METRICS
            ]
            available = [value for value in changes if value is not None]
            improved = sum(value < 0 for value in available)
            worsened = sum(value > 0 for value in available)
            scenario_rows.append(
                {
                    "model": label,
                    "sample_id": sample_id,
                    "components_improved": improved,
                    "components_worsened": worsened,
                    "components_available": len(available),
                    "majority_components_improved": int(improved >= 3),
                    **{metric: value for metric, value in zip(METRICS, changes)},
                }
            )
        model_scenarios = [row for row in scenario_rows if row["model"] == label]
        summary.append(
            {
                "model": label,
                "metric": "MAJORITY_COMPONENTS_IMPROVED",
                "n_scenarios": len(model_scenarios),
                "n_residual_scenarios": "",
                "mean_mitigated_minus_original_goal": "",
                "direct_paired_bootstrap_ci95_low": "",
                "direct_paired_bootstrap_ci95_high": "",
                "scenarios_moving_toward_neutral_rate": mean(
                    float(row["majority_components_improved"]) for row in model_scenarios
                ),
                "mean_mitigated_minus_neutral": "",
            }
        )
    write_csv(output_dir / "mitigation_summary.csv", summary)
    write_csv(output_dir / "mitigation_scenario_changes.csv", scenario_rows)
    print(f"Analyzed {len(runs)} mitigation runs. Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("MODEL", "DIRECT_CORE_CSV", "RESIDUAL_CORE_CSV"),
        required=True,
    )
    parser.add_argument("--output-dir", default="analysis/mitigation")
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    analyze(args.run, Path(args.output_dir), args.seed)


if __name__ == "__main__":
    main()

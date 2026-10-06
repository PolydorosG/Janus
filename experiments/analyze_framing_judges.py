#!/usr/bin/env python3
"""Compare original Qwen framing judgments with a cross-family judge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PairKey = tuple[str, str, str, int, str]


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict) and value.get("score") in {-1, 0, 1}:
                    yield value


def pair_key(row: dict[str, Any]) -> PairKey:
    return (
        str(row.get("sample_id")),
        str(row.get("condition")),
        str(row.get("item")),
        int(row.get("unit_id")),
        str(row.get("fact_id")),
    )


def model_name(path: Path, suffix: str) -> str:
    name = path.name
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def family_group(model: str) -> str:
    return "qwen_evaluated" if model.lower().startswith("qwen") else "non_qwen_evaluated"


def mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else None


def standard_error(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = sum(values) / len(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def pearson_correlation(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = list(left)
    y = list(right)
    if len(x) != len(y) or len(x) < 2:
        return None
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_scale = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_scale = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    denominator = x_scale * y_scale
    return numerator / denominator if denominator else None


def ranks(values: Iterable[float]) -> list[float]:
    numbers = list(values)
    ordered = sorted(range(len(numbers)), key=numbers.__getitem__)
    result = [0.0] * len(numbers)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and numbers[ordered[end]] == numbers[ordered[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            result[index] = average_rank
        start = end
    return result


def spearman_correlation(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = list(left)
    y = list(right)
    return pearson_correlation(ranks(x), ranks(y))


def direction(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def bootstrap_mean_ci(
    values: list[float], seed: int = 20260710, iterations: int = 10000
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(iterations)
    )
    return estimates[int(0.025 * iterations)], estimates[int(0.975 * iterations)]


def cohen_kappa(pairs: list[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    labels = (-1, 0, 1)
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        left_counts[label] / len(pairs) * right_counts[label] / len(pairs)
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as output_file:
        if fields:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def compare(
    original_dir: str | Path,
    alternate_dir: str | Path,
    alternate_tag: str,
    output_dir: str | Path,
) -> None:
    original_root = Path(original_dir)
    alternate_root = Path(alternate_dir)
    output_root = Path(output_dir)
    pair_rows: list[dict[str, Any]] = []
    sample_condition_scores: dict[
        tuple[str, str, str, str], dict[str, list[int]]
    ] = defaultdict(lambda: {"original": [], "alternate": []})

    alternate_suffix = f"_{alternate_tag}.jsonl"
    for alternate_path in sorted(alternate_root.glob(f"*{alternate_suffix}")):
        model = model_name(alternate_path, alternate_suffix)
        original_path = original_root / f"{model}_framing.jsonl"
        if not original_path.exists():
            print(f"Skipping {model}: missing {original_path}")
            continue
        original = {pair_key(row): row for row in read_jsonl(original_path)}
        alternate = {pair_key(row): row for row in read_jsonl(alternate_path)}
        for key in sorted(original.keys() & alternate.keys()):
            original_score = int(original[key]["score"])
            alternate_score = int(alternate[key]["score"])
            sample_id, condition, item, unit_id, fact_id = key
            family = family_group(model)
            pair_rows.append(
                {
                    "model": model,
                    "evaluated_family_group": family,
                    "sample_id": sample_id,
                    "condition": condition,
                    "item": item,
                    "unit_id": unit_id,
                    "fact_id": fact_id,
                    "original_score": original_score,
                    "alternate_score": alternate_score,
                    "exact_agreement": int(original_score == alternate_score),
                    "alternate_minus_original": alternate_score - original_score,
                }
            )
            group_key = (model, sample_id, condition, item)
            sample_condition_scores[group_key]["original"].append(original_score)
            sample_condition_scores[group_key]["alternate"].append(alternate_score)

    condition_means: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for (model, sample_id, condition, _item), scores in sample_condition_scores.items():
        condition_means[(model, sample_id, condition)] = {
            "original": {"mean": float(mean(scores["original"]) or 0)},
            "alternate": {"mean": float(mean(scores["alternate"]) or 0)},
        }

    delta_rows: list[dict[str, Any]] = []
    sample_keys = {(model, sample_id) for model, sample_id, _ in condition_means}
    for model, sample_id in sorted(sample_keys):
        neutral = condition_means.get((model, sample_id, "neutral"))
        goal = condition_means.get((model, sample_id, "latent_goal"))
        if neutral is None or goal is None:
            continue
        original_delta = goal["original"]["mean"] - neutral["original"]["mean"]
        alternate_delta = goal["alternate"]["mean"] - neutral["alternate"]["mean"]
        delta_rows.append(
            {
                "model": model,
                "evaluated_family_group": family_group(model),
                "sample_id": sample_id,
                "original_framing_delta": original_delta,
                "alternate_framing_delta": alternate_delta,
                "alternate_minus_original_delta": alternate_delta - original_delta,
            }
        )

    deltas_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        deltas_by_model[str(row["model"])].append(row)
    model_rows: list[dict[str, Any]] = []
    for model, rows in sorted(deltas_by_model.items()):
        original_mean = mean(float(row["original_framing_delta"]) for row in rows)
        alternate_mean = mean(float(row["alternate_framing_delta"]) for row in rows)
        if original_mean is None or alternate_mean is None:
            continue
        model_rows.append(
            {
                "model": model,
                "evaluated_family_group": family_group(model),
                "n_scenarios": len(rows),
                "mean_original_framing_delta": original_mean,
                "mean_alternate_framing_delta": alternate_mean,
                "alternate_minus_original_delta": alternate_mean - original_mean,
                "original_direction": direction(original_mean),
                "alternate_direction": direction(alternate_mean),
                "direction_agreement": int(direction(original_mean) == direction(alternate_mean)),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for group in ("ALL", "qwen_evaluated", "non_qwen_evaluated"):
        selected_pairs = pair_rows if group == "ALL" else [
            row for row in pair_rows if row["evaluated_family_group"] == group
        ]
        selected_deltas = delta_rows if group == "ALL" else [
            row for row in delta_rows if row["evaluated_family_group"] == group
        ]
        score_pairs = [
            (int(row["original_score"]), int(row["alternate_score"]))
            for row in selected_pairs
        ]
        differences = [
            float(row["alternate_minus_original_delta"]) for row in selected_deltas
        ]
        se = standard_error(differences)
        difference_mean = mean(differences)
        summary_rows.append(
            {
                "evaluated_family_group": group,
                "n_pairs": len(selected_pairs),
                "exact_agreement": mean(float(row["exact_agreement"]) for row in selected_pairs),
                "cohen_kappa": cohen_kappa(score_pairs),
                "mean_original_score": mean(float(row["original_score"]) for row in selected_pairs),
                "mean_alternate_score": mean(float(row["alternate_score"]) for row in selected_pairs),
                "n_sample_deltas": len(selected_deltas),
                "mean_original_framing_delta": mean(float(row["original_framing_delta"]) for row in selected_deltas),
                "mean_alternate_framing_delta": mean(float(row["alternate_framing_delta"]) for row in selected_deltas),
                "mean_alternate_minus_original_delta": difference_mean,
                "difference_ci95_low": None if difference_mean is None or se is None else difference_mean - 1.96 * se,
                "difference_ci95_high": None if difference_mean is None or se is None else difference_mean + 1.96 * se,
                "pair_score_pearson": pearson_correlation(
                    (float(row["original_score"]) for row in selected_pairs),
                    (float(row["alternate_score"]) for row in selected_pairs),
                ),
                "pair_score_spearman": spearman_correlation(
                    (float(row["original_score"]) for row in selected_pairs),
                    (float(row["alternate_score"]) for row in selected_pairs),
                ),
                "scenario_delta_pearson": pearson_correlation(
                    (float(row["original_framing_delta"]) for row in selected_deltas),
                    (float(row["alternate_framing_delta"]) for row in selected_deltas),
                ),
                "scenario_delta_spearman": spearman_correlation(
                    (float(row["original_framing_delta"]) for row in selected_deltas),
                    (float(row["alternate_framing_delta"]) for row in selected_deltas),
                ),
            }
        )

    qwen_differences = [
        float(row["alternate_minus_original_delta"])
        for row in delta_rows
        if row["evaluated_family_group"] == "qwen_evaluated"
    ]
    non_qwen_differences = [
        float(row["alternate_minus_original_delta"])
        for row in delta_rows
        if row["evaluated_family_group"] == "non_qwen_evaluated"
    ]
    if qwen_differences and non_qwen_differences:
        interaction = mean(qwen_differences) - mean(non_qwen_differences)  # type: ignore[operator]
        interaction_se = math.sqrt(
            (standard_error(qwen_differences) or 0) ** 2
            + (standard_error(non_qwen_differences) or 0) ** 2
        )
        summary_rows.append(
            {
                "evaluated_family_group": "QWEN_MINUS_NON_QWEN_INTERACTION",
                "n_pairs": "",
                "exact_agreement": "",
                "cohen_kappa": "",
                "mean_original_score": "",
                "mean_alternate_score": "",
                "n_sample_deltas": len(qwen_differences) + len(non_qwen_differences),
                "mean_original_framing_delta": "",
                "mean_alternate_framing_delta": "",
                "mean_alternate_minus_original_delta": interaction,
                "difference_ci95_low": interaction - 1.96 * interaction_se,
                "difference_ci95_high": interaction + 1.96 * interaction_se,
                "pair_score_pearson": "",
                "pair_score_spearman": "",
                "scenario_delta_pearson": "",
                "scenario_delta_spearman": "",
            }
        )

    model_original = [float(row["mean_original_framing_delta"]) for row in model_rows]
    model_alternate = [float(row["mean_alternate_framing_delta"]) for row in model_rows]
    direction_matches = sum(int(row["direction_agreement"]) for row in model_rows)
    cross_judge_rows = [
        {
            "n_models": len(model_rows),
            "model_delta_pearson": pearson_correlation(model_original, model_alternate),
            "model_delta_spearman": spearman_correlation(model_original, model_alternate),
            "direction_agreement_models": direction_matches,
            "direction_agreement_rate": (
                direction_matches / len(model_rows) if model_rows else None
            ),
            "non_qwen_original_mean_delta": mean(
                float(row["mean_original_framing_delta"])
                for row in model_rows
                if row["evaluated_family_group"] == "non_qwen_evaluated"
            ),
            "non_qwen_alternate_mean_delta": mean(
                float(row["mean_alternate_framing_delta"])
                for row in model_rows
                if row["evaluated_family_group"] == "non_qwen_evaluated"
            ),
        }
    ]

    qwen_model = "qwen3_8b_no_thinking"
    llama_model = "llama_3_1_8b_instruct"
    by_model_sample = {
        (str(row["model"]), str(row["sample_id"])): row for row in delta_rows
    }
    shared_samples = sorted(
        sample_id
        for model, sample_id in by_model_sample
        if model == qwen_model and (llama_model, sample_id) in by_model_sample
    )
    interaction_values = [
        float(by_model_sample[(qwen_model, sample_id)]["alternate_minus_original_delta"])
        - float(by_model_sample[(llama_model, sample_id)]["alternate_minus_original_delta"])
        for sample_id in shared_samples
    ]
    interaction_low, interaction_high = bootstrap_mean_ci(interaction_values)

    def model_delta_mean(model: str, field: str) -> float | None:
        return mean(
            float(row[field]) for row in delta_rows if str(row["model"]) == model
        )

    qwen_original = model_delta_mean(qwen_model, "original_framing_delta")
    qwen_alternate = model_delta_mean(qwen_model, "alternate_framing_delta")
    llama_original = model_delta_mean(llama_model, "original_framing_delta")
    llama_alternate = model_delta_mean(llama_model, "alternate_framing_delta")
    family_bias_rows = [
        {
            "n_shared_scenarios": len(shared_samples),
            "qwen_generator_original_judge_delta": qwen_original,
            "qwen_generator_alternate_judge_delta": qwen_alternate,
            "qwen_generator_judge_change": (
                None if qwen_original is None or qwen_alternate is None else qwen_alternate - qwen_original
            ),
            "llama_generator_original_judge_delta": llama_original,
            "llama_generator_alternate_judge_delta": llama_alternate,
            "llama_generator_judge_change": (
                None if llama_original is None or llama_alternate is None else llama_alternate - llama_original
            ),
            "qwen_minus_llama_judge_change": mean(interaction_values),
            "paired_bootstrap_ci95_low": interaction_low,
            "paired_bootstrap_ci95_high": interaction_high,
            "qwen_direction_agreement": (
                None if qwen_original is None or qwen_alternate is None else int(direction(qwen_original) == direction(qwen_alternate))
            ),
            "llama_direction_agreement": (
                None if llama_original is None or llama_alternate is None else int(direction(llama_original) == direction(llama_alternate))
            ),
        }
    ]

    write_csv(output_root / "pair_level_comparison.csv", pair_rows)
    write_csv(output_root / "sample_delta_comparison.csv", delta_rows)
    write_csv(output_root / "model_level_comparison.csv", model_rows)
    write_csv(output_root / "judge_robustness_summary.csv", summary_rows)
    write_csv(output_root / "cross_judge_model_summary.csv", cross_judge_rows)
    write_csv(output_root / "two_generator_family_bias.csv", family_bias_rows)
    print(
        f"Compared {len(pair_rows)} matched framing pairs and {len(delta_rows)} "
        f"sample deltas. Output: {output_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", default="results")
    parser.add_argument("--alternate-dir", default="analysis/framing_llama")
    parser.add_argument("--alternate-tag", default="framing_llama")
    parser.add_argument("--output-dir", default="analysis/judge_robustness")
    args = parser.parse_args()
    compare(
        args.original_dir,
        args.alternate_dir,
        args.alternate_tag,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

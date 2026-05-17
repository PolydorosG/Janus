"""Create per-model radar charts from Janus core metrics."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


FAMILY_ORDER = ["OpenAI", "DeepSeek", "Llama", "Qwen", "Other"]

MODEL_FAMILIES = {
    "deepseek_chat": "DeepSeek",
    "deepseek_reasoner": "DeepSeek",
    "gpt_5_4": "OpenAI",
    "gpt_5_4_mini": "OpenAI",
    "llama_3_1_8b_instruct": "Llama",
    "llama_3_1_70b_instruct": "Llama",
    "qwen3_8b_no_thinking": "Qwen",
    "qwen3_8b_thinking": "Qwen",
    "qwen3_14b_no_thinking": "Qwen",
    "qwen3_14b_thinking": "Qwen",
    "qwen3_32b_no_thinking": "Qwen",
    "qwen3_32b_thinking": "Qwen",
}

FAMILY_COLORS = {
    "OpenAI": "#2563eb",
    "DeepSeek": "#dc2626",
    "Llama": "#059669",
    "Qwen": "#d97706",
    "Other": "#6b7280",
}

MODEL_COLORS = {
    "gpt_5_4": "#1f77b4",
    "gpt_5_4_mini": "#aec7e8",
    "deepseek_chat": "#d62728",
    "deepseek_reasoner": "#ff9896",
    "llama_3_1_70b_instruct": "#2ca02c",
    "llama_3_1_8b_instruct": "#98df8a",
    "qwen3_14b_no_thinking": "#ff7f0e",
    "qwen3_14b_thinking": "#ffbb78",
    "qwen3_32b_no_thinking": "#9467bd",
    "qwen3_32b_thinking": "#c5b0d5",
    "qwen3_8b_no_thinking": "#8c564b",
    "qwen3_8b_thinking": "#c49c94",
}

DISPLAY_NAMES = {
    "deepseek_chat": "DeepSeek Chat",
    "deepseek_reasoner": "DeepSeek Reasoner",
    "gpt_5_4": "GPT-5.4",
    "gpt_5_4_mini": "GPT-5.4 Mini",
    "llama_3_1_8b_instruct": "Llama 3.1 8B Instruct",
    "llama_3_1_70b_instruct": "Llama 3.1 70B Instruct",
    "qwen3_8b_no_thinking": "Qwen3 8B No Thinking",
    "qwen3_8b_thinking": "Qwen3 8B Thinking",
    "qwen3_14b_no_thinking": "Qwen3 14B No Thinking",
    "qwen3_14b_thinking": "Qwen3 14B Thinking",
    "qwen3_32b_no_thinking": "Qwen3 32B No Thinking",
    "qwen3_32b_thinking": "Qwen3 32B Thinking",
}

MODEL_SIZES_B = {
    "deepseek_chat": 671.0,
    "deepseek_reasoner": 671.0,
    "llama_3_1_8b_instruct": 8.0,
    "llama_3_1_70b_instruct": 70.0,
    "qwen3_8b_no_thinking": 8.0,
    "qwen3_8b_thinking": 8.0,
    "qwen3_14b_no_thinking": 14.0,
    "qwen3_14b_thinking": 14.0,
    "qwen3_32b_no_thinking": 32.0,
    "qwen3_32b_thinking": 32.0,
}

METRICS = [
    ("delta_selection_gap", "Selection"),
    ("delta_emphasis_gap", "Emphasis"),
    ("delta_ordering_shift", "Ordering"),
    ("delta_specificity", "Specificity"),
    ("framing_delta", "Framing"),
]

ABSOLUTE_METRICS = [
    ("selection_bias", "Selection"),
    ("emphasis_bias", "Emphasis"),
    ("ordering_bias", "Ordering"),
    ("specificity_loss", "Specificity"),
    ("framing_bias", "Framing"),
]

NATURAL_SCALES = {
    "delta_selection_gap": 2.0,
    "delta_emphasis_gap": 2.0,
    "delta_ordering_shift": 1.0,
    "delta_specificity": 1.0,
    "framing_delta": 2.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one radar chart per model from core_metrics.csv files."
    )
    parser.add_argument(
        "--metrics-root",
        default="metrics",
        help="Directory containing per-model metric folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="metrics/radar_charts",
        help="Directory for radar PNGs and model summary CSV.",
    )
    args = parser.parse_args()

    metrics_root = Path(args.metrics_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_means = load_model_means(metrics_root)
    if not model_means:
        raise SystemExit(f"No core_metrics.csv files found under {metrics_root}")

    category_means = load_category_means(metrics_root)
    apply_pairwise_ordering_metrics(metrics_root, model_means, category_means)
    scales = metric_scales(model_means.values())
    visual_scales = observed_metric_scales(model_means.values())
    write_summary(output_dir / "model_core_metric_means.csv", model_means, scales)
    write_latex_table(
        output_dir / "model_core_metric_means_table.tex",
        model_means=model_means,
        scales=scales,
    )
    write_category_composite_summary(
        output_dir / "category_composite_scores.csv",
        category_means=category_means,
        scales=scales,
    )
    save_composite_bar_chart(
        output_dir / "composite_distortion_scores.png",
        model_means=model_means,
        scales=scales,
    )
    save_size_ordered_chart(
        output_dir / "composite_by_model_size.png",
        model_means=model_means,
        scales=scales,
    )
    save_category_composite_radar_chart(
        output_dir / "category_composite_radar.png",
        category_means=category_means,
        scales=scales,
    )
    save_category_composite_heatmap(
        output_dir / "category_composite_heatmap.png",
        category_means=category_means,
        scales=scales,
    )
    save_metric_delta_heatmap(
        output_dir / "model_metric_delta_heatmap.png",
        model_means=model_means,
        scales=visual_scales,
    )
    condition_means = load_condition_absolute_means(metrics_root)
    save_condition_overlay_radar_grid(
        output_dir / "neutral_goal_absolute_radar_overlay.png",
        condition_means=condition_means,
    )
    save_condition_overlay_radar_grid(
        output_dir / "neutral_goal_absolute_radar_overlay_axis_normalized.png",
        condition_means=condition_means,
        axis_normalized=True,
    )

    for model, means in sorted(model_means.items()):
        save_radar_chart(
            output_dir / f"{slugify(model)}_radar.png",
            model=model,
            means=means,
            scales=visual_scales,
        )
    save_joint_radar_chart(
        output_dir / "all_models_radar.png",
        model_means=model_means,
        scales=visual_scales,
    )

    print(
        f"Wrote {len(model_means)} per-model radar charts, one joint chart, "
        f"one category radar, one composite score chart, one category heatmap, "
        f"one size-ordered chart, two neutral/goal overlay charts, and one "
        f"LaTeX table to {output_dir}"
    )


def load_model_means(metrics_root: Path) -> dict[str, dict[str, float]]:
    model_means: dict[str, dict[str, float]] = {}
    for core_path in sorted(metrics_root.glob("*/core_metrics.csv")):
        model = core_path.parent.name
        rows = list(read_csv(core_path))
        if not rows:
            continue
        means: dict[str, float] = {}
        for metric, _ in METRICS:
            values = [parse_float(row.get(metric, "")) for row in rows]
            numeric_values = [value for value in values if value is not None]
            means[metric] = mean(numeric_values) if numeric_values else 0.0
        model_means[model] = means
    return model_means


def load_category_means(
    metrics_root: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    category_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    for core_path in sorted(metrics_root.glob("*/core_metrics.csv")):
        model = core_path.parent.name
        for row in read_csv(core_path):
            category = row.get("category") or "Uncategorized"
            metrics_for_category = category_values.setdefault(model, {}).setdefault(
                category,
                {metric: [] for metric, _ in METRICS},
            )
            for metric, _ in METRICS:
                value = parse_float(row.get(metric, ""))
                if value is not None:
                    metrics_for_category[metric].append(value)

    category_means: dict[str, dict[str, dict[str, float]]] = {}
    for model, category_rows in category_values.items():
        category_means[model] = {}
        for category, values_by_metric in category_rows.items():
            category_means[model][category] = {
                metric: mean(values) if values else 0.0
                for metric, values in values_by_metric.items()
            }
    return category_means


def load_condition_absolute_means(
    metrics_root: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    condition_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    for output_path in sorted(metrics_root.glob("*/output_level_metrics.csv")):
        model = output_path.parent.name
        for row in read_csv(output_path):
            condition = row.get("condition") or ""
            if condition not in {"neutral", "latent_goal"}:
                continue
            metric_values = absolute_metric_values(row)
            metrics_for_condition = condition_values.setdefault(model, {}).setdefault(
                condition,
                {metric: [] for metric, _ in ABSOLUTE_METRICS},
            )
            for metric, value in metric_values.items():
                if value is not None:
                    metrics_for_condition[metric].append(value)

    condition_means: dict[str, dict[str, dict[str, float]]] = {}
    for model, condition_rows in condition_values.items():
        condition_means[model] = {}
        for condition, values_by_metric in condition_rows.items():
            condition_means[model][condition] = {
                metric: mean(values) if values else 0.0
                for metric, values in values_by_metric.items()
            }
    apply_absolute_pairwise_ordering(metrics_root, condition_means)
    return condition_means


def absolute_metric_values(row: dict[str, str]) -> dict[str, float | None]:
    selection_gap = parse_float(row.get("selection_gap"))
    emphasis_gap = parse_float(row.get("emphasis_gap"))
    specificity = parse_float(row.get("specificity"))
    framing_raw = parse_float(row.get("framing_raw"))
    return {
        "selection_bias": positive_part(selection_gap),
        "emphasis_bias": positive_part(emphasis_gap),
        "specificity_loss": None if specificity is None else max(0.0, 1.0 - specificity),
        "framing_bias": positive_part(framing_raw),
    }


def apply_absolute_pairwise_ordering(
    metrics_root: Path,
    condition_means: dict[str, dict[str, dict[str, float]]],
) -> None:
    for fact_path in sorted(metrics_root.glob("*/fact_level_metrics.csv")):
        model = fact_path.parent.name
        values_by_condition: dict[str, list[float]] = {"neutral": [], "latent_goal": []}
        for (_, condition), rows in grouped_fact_rows(fact_path).items():
            if condition in values_by_condition:
                values_by_condition[condition].append(
                    1.0 - adverse_first_pair_rate(rows)
                )
        for condition, values in values_by_condition.items():
            if values and model in condition_means and condition in condition_means[model]:
                condition_means[model][condition]["ordering_bias"] = mean(values)


def positive_part(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, value)


def apply_pairwise_ordering_metrics(
    metrics_root: Path,
    model_means: dict[str, dict[str, float]],
    category_means: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Replace raw mean-position ordering with normalized pairwise ordering.

    Positive values mean the latent-goal output has fewer adverse-before-favorable
    pairs than the neutral output, i.e. favorable facts are moved earlier relative
    to adverse facts.
    """

    for fact_path in sorted(metrics_root.glob("*/fact_level_metrics.csv")):
        model = fact_path.parent.name
        output_scores: dict[tuple[str, str], dict[str, object]] = {}
        for key, rows in grouped_fact_rows(fact_path).items():
            output_scores[key] = {
                "category": rows[0].get("category") or "Uncategorized",
                "adverse_first_rate": adverse_first_pair_rate(rows),
            }

        deltas_by_category: dict[str, list[float]] = {}
        sample_ids = sorted({sample_id for sample_id, _ in output_scores})
        model_deltas: list[float] = []
        for sample_id in sample_ids:
            neutral = output_scores.get((sample_id, "neutral"))
            goal = output_scores.get((sample_id, "latent_goal"))
            if neutral is None or goal is None:
                continue
            delta = float(neutral["adverse_first_rate"]) - float(goal["adverse_first_rate"])
            category = str(goal["category"] or neutral["category"] or "Uncategorized")
            model_deltas.append(delta)
            deltas_by_category.setdefault(category, []).append(delta)

        if model_deltas and model in model_means:
            model_means[model]["delta_ordering_shift"] = mean(model_deltas)
        for category, deltas in deltas_by_category.items():
            if model in category_means and category in category_means[model]:
                category_means[model][category]["delta_ordering_shift"] = mean(deltas)


def grouped_fact_rows(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in read_csv(path):
        sample_id = row.get("sample_id") or ""
        condition = row.get("condition") or ""
        grouped.setdefault((sample_id, condition), []).append(row)
    return grouped


def adverse_first_pair_rate(rows: list[dict[str, str]]) -> float:
    favorable_positions = [
        value
        for value in (
            parse_float(row.get("first_output_unit_id"))
            for row in rows
            if row.get("polarity") == "F"
        )
        if value is not None
    ]
    adverse_positions = [
        value
        for value in (
            parse_float(row.get("first_output_unit_id"))
            for row in rows
            if row.get("polarity") == "A"
        )
        if value is not None
    ]
    adverse_first_pairs = sum(
        1
        for favorable_position in favorable_positions
        for adverse_position in adverse_positions
        if adverse_position < favorable_position
    )
    return adverse_first_pairs / 9.0


def read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        yield from csv.DictReader(csv_file)


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def metric_scales(model_means: Iterable[dict[str, float]]) -> dict[str, float]:
    return {metric: NATURAL_SCALES[metric] for metric, _ in METRICS}


def observed_metric_scales(model_means: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(model_means)
    return {
        metric: max(max(abs(row.get(metric, 0.0)) for row in rows), 1e-12)
        for metric, _ in METRICS
    }


def write_summary(
    path: Path,
    model_means: dict[str, dict[str, float]],
    scales: dict[str, float],
) -> None:
    fieldnames = [
        "model",
        "display_model",
        *[metric for metric, _ in METRICS],
        "composite_signed",
        "composite_signed_0_100",
        "composite_distortion_0_100",
        *[f"{metric}_plot_value" for metric, _ in METRICS],
        *[f"{metric}_normalized" for metric, _ in METRICS],
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for model, means in sorted(model_means.items()):
            normalized = normalized_metric_values(means, scales)
            signed = mean(normalized)
            distortion = mean([max(0.0, value) for value in normalized])
            row = {
                "model": model,
                "display_model": display_name(model),
                "composite_signed": f"{signed:.12g}",
                "composite_signed_0_100": f"{50.0 * (signed + 1.0):.12g}",
                "composite_distortion_0_100": f"{100.0 * distortion:.12g}",
            }
            for metric, _ in METRICS:
                row[metric] = f"{means[metric]:.12g}"
                row[f"{metric}_plot_value"] = f"{plot_value(means[metric], scales[metric]):.12g}"
                row[f"{metric}_normalized"] = f"{normalized_metric(means[metric], scales[metric]):.12g}"
            writer.writerow(row)


def write_latex_table(
    path: Path,
    *,
    model_means: dict[str, dict[str, float]],
    scales: dict[str, float],
) -> None:
    metric_columns = [metric for metric, _ in METRICS]
    columns = [*metric_columns, "composite_distortion_0_100"]
    table_values: dict[str, dict[str, float]] = {}
    for model, means in model_means.items():
        normalized = normalized_metric_values(means, scales)
        table_values[model] = {
            metric: 100.0 * plot_value(means[metric], scales[metric])
            for metric in metric_columns
        }
        table_values[model]["composite_distortion_0_100"] = 100.0 * mean(
            [max(0.0, value) for value in normalized]
        )

    best_by_column = {
        column: top_two_values([values[column] for values in table_values.values()])
        for column in columns
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Model-level Janus metric indices and composite distortion scores. Metric columns are signed 0--100 indices; Composite is the positive-only normalized distortion score.}",
        r"\label{tab:janus-model-core-metrics}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Selection & Emphasis & Ordering & Specificity & Framing & Composite \\",
        r"\midrule",
    ]

    first_family = True
    for family in FAMILY_ORDER:
        models = [model for model in sorted_models(model_means) if model_family(model) == family]
        if not models:
            continue
        if not first_family:
            lines.append(r"\midrule")
        first_family = False
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{escape_latex(family)}}}}} \\")
        for model in models:
            cells = [escape_latex(display_name(model))]
            for column in columns:
                cells.append(format_latex_score(table_values[model][column], best_by_column[column]))
            lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def top_two_values(values: list[float]) -> tuple[float | None, float | None]:
    unique_values: list[float] = []
    for value in sorted(values, reverse=True):
        if not any(math.isclose(value, existing, rel_tol=1e-9, abs_tol=1e-9) for existing in unique_values):
            unique_values.append(value)
        if len(unique_values) == 2:
            break
    best = unique_values[0] if unique_values else None
    second = unique_values[1] if len(unique_values) > 1 else None
    return best, second


def format_latex_score(value: float, best_second: tuple[float | None, float | None]) -> str:
    formatted = f"{value:.1f}"
    best, second = best_second
    if best is not None and math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-9):
        return rf"\textbf{{{formatted}}}"
    if second is not None and math.isclose(value, second, rel_tol=1e-9, abs_tol=1e-9):
        return rf"\underline{{{formatted}}}"
    return formatted


def escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_category_composite_summary(
    path: Path,
    *,
    category_means: dict[str, dict[str, dict[str, float]]],
    scales: dict[str, float],
) -> None:
    fieldnames = ["model", "display_model", "family", "category", "composite_distortion_0_100"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for model in sorted_models(category_means):
            for category in sorted(category_means[model]):
                writer.writerow(
                    {
                        "model": model,
                        "display_model": display_name(model),
                        "family": model_family(model),
                        "category": category,
                        "composite_distortion_0_100": f"{category_composite_score(category_means[model][category], scales):.12g}",
                    }
                )


def plot_value(value: float, scale: float) -> float:
    return 0.5 + 0.5 * normalized_metric(value, scale)


def normalized_metric(value: float, scale: float) -> float:
    clipped = max(-1.0, min(1.0, value / scale))
    return clipped


def normalized_metric_values(
    means: dict[str, float],
    scales: dict[str, float],
) -> list[float]:
    return [
        normalized_metric(means[metric], scales[metric])
        for metric, _ in METRICS
    ]


def composite_distortion_score(
    means: dict[str, float],
    scales: dict[str, float],
) -> float:
    normalized = normalized_metric_values(means, scales)
    return 100.0 * mean([max(0.0, value) for value in normalized])


def category_composite_score(
    means: dict[str, float],
    scales: dict[str, float],
) -> float:
    return composite_distortion_score(means, scales)


def save_radar_chart(
    path: Path,
    *,
    model: str,
    means: dict[str, float],
    scales: dict[str, float],
) -> None:
    labels = [label for _, label in METRICS]
    angles = radar_angles()
    values = [plot_value(means[metric], scales[metric]) for metric, _ in METRICS]
    angles_closed = angles + angles[:1]
    values_closed = values + values[:1]

    fig = plt.figure(figsize=(7.2, 7.2), dpi=180)
    ax = fig.add_subplot(111, polar=True)
    style_radar_axis(ax, angles, labels)
    color = model_color(model, 0)
    ax.plot(
        angles_closed,
        values_closed,
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=4.0,
    )
    ax.fill(angles_closed, values_closed, color=color, alpha=0.18)
    ax.plot(angles_closed, [0.5] * len(angles_closed), color="#343a40", linewidth=1.0)

    ax.set_title(display_name(model), va="bottom", fontsize=14, pad=24)
    # ax.text(
    #     0.5,
    #     -0.11,
    #     "Axis radii are scaled to observed model range per metric.",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="center",
    #     fontsize=7.5,
    #     color="#4b5259",
    # )
    fig.tight_layout(rect=(0.03, 0.03, 0.97, 0.95))
    fig.savefig(path)
    plt.close(fig)


def save_joint_radar_chart(
    path: Path,
    *,
    model_means: dict[str, dict[str, float]],
    scales: dict[str, float],
) -> None:
    labels = [label for _, label in METRICS]
    angles = radar_angles()
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(10.6, 8.2), dpi=180)
    ax = fig.add_subplot(111, polar=True)
    style_radar_axis(ax, angles, labels)
    ax.plot(angles_closed, [0.5] * len(angles_closed), color="#444b52", linewidth=1.0)

    for index, model in enumerate(sorted_models(model_means)):
        means = model_means[model]
        values = [plot_value(means[metric], scales[metric]) for metric, _ in METRICS]
        values_closed = values + values[:1]
        color = model_color(model, index)
        ax.plot(
            angles_closed,
            values_closed,
            color=color,
            linewidth=1.7,
            alpha=0.92,
            marker="o",
            markersize=3.0,
            label=display_name(model),
        )
        ax.fill(angles_closed, values_closed, color=color, alpha=0.04)

    ax.set_title("All Models", va="bottom", fontsize=15, pad=24)
    # ax.text(
    #     0.5,
    #     -0.11,
    #     "Axis radii are scaled to observed model range per metric.",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="center",
    #     fontsize=8,
    #     color="#4b5259",
    # )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.08, 0.5),
        frameon=False,
        fontsize=8.5,
        labelspacing=0.72,
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.78, 0.95))
    fig.savefig(path)
    plt.close(fig)


def save_condition_overlay_radar_grid(
    path: Path,
    *,
    condition_means: dict[str, dict[str, dict[str, float]]],
    axis_normalized: bool = False,
) -> None:
    models = sorted_models(condition_means)
    labels = [label for _, label in ABSOLUTE_METRICS]
    angles = [2 * math.pi * index / len(ABSOLUTE_METRICS) for index in range(len(ABSOLUTE_METRICS))]
    angles_closed = angles + angles[:1]
    condition_styles = {
        "neutral": ("Neutral", "#2563eb", "-"),
        "latent_goal": ("Goal-conditioned", "#dc2626", "-"),
    }
    axis_scales = absolute_axis_scales(condition_means) if axis_normalized else {
        metric: 1.0 for metric, _ in ABSOLUTE_METRICS
    }

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14.2, 10.4),
        dpi=180,
        subplot_kw={"polar": True},
    )

    for ax, model in zip(axes.flat, models):
        style_absolute_radar_axis(ax, angles, labels)
        for condition, (label, color, linestyle) in condition_styles.items():
            means = condition_means.get(model, {}).get(condition)
            if means is None:
                continue
            values = [
                max(0.0, min(1.0, means.get(metric, 0.0) / axis_scales[metric]))
                for metric, _ in ABSOLUTE_METRICS
            ]
            values_closed = values + values[:1]
            ax.plot(
                angles_closed,
                values_closed,
                color=color,
                linewidth=1.8,
                linestyle=linestyle,
                marker="o",
                markersize=2.8,
                alpha=0.96,
                label=label,
            )
            ax.fill(angles_closed, values_closed, color=color, alpha=0.08)
        ax.set_title(display_name(model), va="bottom", fontsize=10.2, pad=13)

    for ax in axes.flat[len(models):]:
        ax.axis("off")

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        (
            "Axis-Normalized Absolute Distortion Profiles by Prompt Condition"
            if axis_normalized
            else "Absolute Distortion Profiles by Prompt Condition"
        ),
        fontsize=15,
        y=0.985,
    )
    note = (
        "Each axis is divided by that metric's observed maximum across all model-condition pairs; shapes emphasize relative within-metric differences."
        if axis_normalized
        else "Radii show condition-level absolute distortion scores: positive selection/emphasis bias, favorable-first ordering, numeric specificity loss, and positive framing."
    )
    fig.text(
        0.5,
        0.018,
        note,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4b5259",
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.93), h_pad=2.0, w_pad=1.0)
    fig.savefig(path)
    plt.close(fig)


def absolute_axis_scales(
    condition_means: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for metric, _ in ABSOLUTE_METRICS:
        max_value = max(
            (
                condition_rows[condition].get(metric, 0.0)
                for condition_rows in condition_means.values()
                for condition in condition_rows
            ),
            default=0.0,
        )
        scales[metric] = max(max_value, 1e-12)
    return scales


def save_category_composite_radar_chart(
    path: Path,
    *,
    category_means: dict[str, dict[str, dict[str, float]]],
    scales: dict[str, float],
) -> None:
    categories = sorted({category for rows in category_means.values() for category in rows})
    labels = [short_category_label(category) for category in categories]
    angles = [2 * math.pi * index / len(categories) for index in range(len(categories))]
    angles_closed = angles + angles[:1]
    raw_scores_by_model = {
        model: [
            category_composite_score(
                category_means[model].get(category, zero_metric_row()),
                scales,
            )
            for category in categories
        ]
        for model in sorted_models(category_means)
    }
    max_score = max(max(scores) for scores in raw_scores_by_model.values()) or 1.0

    fig = plt.figure(figsize=(11.2, 8.4), dpi=180)
    ax = fig.add_subplot(111, polar=True)
    style_relative_score_radar_axis(ax, angles, labels)

    for index, model in enumerate(sorted_models(category_means)):
        scores = [
            score / max_score
            for score in raw_scores_by_model[model]
        ]
        scores_closed = scores + scores[:1]
        color = model_color(model, index)
        ax.plot(
            angles_closed,
            scores_closed,
            color=color,
            linewidth=1.7,
            alpha=0.92,
            marker="o",
            markersize=3.0,
            label=display_name(model),
        )
        ax.fill(angles_closed, scores_closed, color=color, alpha=0.04)

    ax.set_title("Composite Distortion by Category", va="bottom", fontsize=15, pad=24)
    # ax.text(
    #     0.5,
    #     -0.12,
    #     f"Radii are scaled to observed max category score ({max_score:.1f}).",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="center",
    #     fontsize=8,
    #     color="#4b5259",
    # )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.08, 0.5),
        frameon=False,
        fontsize=8.5,
        labelspacing=0.72,
    )
    fig.tight_layout(rect=(0.03, 0.03, 0.78, 0.95))
    fig.savefig(path)
    plt.close(fig)


def save_category_composite_heatmap(
    path: Path,
    *,
    category_means: dict[str, dict[str, dict[str, float]]],
    scales: dict[str, float],
) -> None:
    models = sorted_models(category_means)
    categories = sorted({category for rows in category_means.values() for category in rows})
    scores = [
        [
            category_composite_score(category_means[model].get(category, zero_metric_row()), scales)
            for category in categories
        ]
        for model in models
    ]

    max_score = max(max(row) for row in scores) or 1.0

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=180)
    image = ax.imshow(scores, cmap="YlOrRd", vmin=0, vmax=max_score, aspect="auto")
    ax.set_xticks(list(range(len(categories))))
    ax.set_xticklabels([short_category_label(category) for category in categories], rotation=35, ha="right")
    ax.set_yticks(list(range(len(models))))
    ax.set_yticklabels([display_name(model) for model in models], fontsize=8.5)
    ax.set_title("Category Composite Distortion Heatmap", fontsize=14, pad=14)
    ax.set_xlabel("Category")
    ax.set_ylabel("Model")

    for row_index, row in enumerate(scores):
        for column_index, score in enumerate(row):
            text_color = "#111827" if score < 0.55 * max_score else "#ffffff"
            ax.text(
                column_index,
                row_index,
                f"{score:.1f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )

    for boundary in family_boundaries(models):
        ax.axhline(boundary - 0.5, color="#ffffff", linewidth=2.0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Composite Distortion Score")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_metric_delta_heatmap(
    path: Path,
    *,
    model_means: dict[str, dict[str, float]],
    scales: dict[str, float],
) -> None:
    models = sorted_models(model_means)
    labels = [label for _, label in METRICS]
    normalized_rows = [
        [
            normalized_metric(model_means[model][metric], scales[metric])
            for metric, _ in METRICS
        ]
        for model in models
    ]
    raw_rows = [
        [model_means[model][metric] for metric, _ in METRICS]
        for model in models
    ]

    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=180)
    image = ax.imshow(normalized_rows, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(list(range(len(models))))
    ax.set_yticklabels([display_name(model) for model in models], fontsize=8.5)
    ax.set_title("Model Metric Deltas", fontsize=14, pad=14)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model")

    for row_index, row in enumerate(raw_rows):
        for column_index, value in enumerate(row):
            normalized = normalized_rows[row_index][column_index]
            text_color = "#ffffff" if abs(normalized) > 0.55 else "#111827"
            ax.text(
                column_index,
                row_index,
                format_heatmap_value(value),
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )

    for boundary in family_boundaries(models):
        ax.axhline(boundary - 0.5, color="#ffffff", linewidth=2.0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Column-normalized signed delta")
    ax.text(
        0.99,
        -0.17,
        "Cell colors are normalized within each metric column; numbers are raw mean deltas.",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=8,
        color="#4b5259",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_composite_bar_chart(
    path: Path,
    *,
    model_means: dict[str, dict[str, float]],
    scales: dict[str, float],
) -> None:
    scores = []
    for model, means in model_means.items():
        distortion = composite_distortion_score(means, scales)
        scores.append((distortion, model))
    scores.sort(reverse=True)

    names = [display_name(model) for _, model in scores]
    distortion_scores = [distortion for distortion, _ in scores]

    fig, ax = plt.subplots(figsize=(9.5, 6.8), dpi=180)
    y_positions = list(range(len(scores)))
    ax.barh(y_positions, distortion_scores, color="#1f6f8b", alpha=0.86)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    max_score = max(distortion_scores) if distortion_scores else 1.0
    ax.set_xlim(0, max_score * 1.18)
    ax.set_xlabel("Composite Distortion Score")
    ax.set_title("Composite Distortion by Model", fontsize=14, pad=14)
    ax.grid(axis="x", color="#d5d9de", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    for y_position, score in zip(y_positions, distortion_scores):
        ax.text(
            score + 1.0,
            y_position,
            f"{score:.1f}",
            va="center",
            fontsize=8,
            color="#22262a",
        )
    ax.text(
        0.99,
        0.02,
        "Bars show the mean of positive naturally normalized metric deltas.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#4b5259",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_size_ordered_chart(
    path: Path,
    *,
    model_means: dict[str, dict[str, float]],
    scales: dict[str, float],
) -> None:
    rows = []
    for model, means in model_means.items():
        distortion = composite_distortion_score(means, scales)
        size_b = MODEL_SIZES_B.get(model)
        rows.append((size_b, distortion, model))

    known_rows = sorted((row for row in rows if row[0] is not None), key=lambda row: row[0])
    unknown_rows = sorted((row for row in rows if row[0] is None), key=lambda row: row[2])

    fig, (ax_known, ax_unknown) = plt.subplots(
        1,
        2,
        figsize=(12.4, 6.6),
        dpi=180,
        gridspec_kw={"width_ratios": [4.2, 1.25]},
    )

    color_by_model = {
        model: model_color(model, index)
        for index, model in enumerate(sorted_models(model_means))
    }

    for size_b, distortion, model in known_rows:
        assert size_b is not None
        ax_known.scatter(
            size_b,
            distortion,
            s=70,
            color=color_by_model[model],
            edgecolor="#20252a",
            linewidth=0.5,
            zorder=3,
        )
        ax_known.annotate(
            display_name(model),
            (size_b, distortion),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7.5,
        )

    ax_known.set_xscale("log")
    ax_known.set_xlim(6, 900)
    max_known_score = max((distortion for _, distortion, _ in known_rows), default=1.0)
    ax_known.set_ylim(0, max_known_score * 1.18)
    ax_known.set_xlabel("Model Size (B parameters, log scale)")
    ax_known.set_ylabel("Composite Distortion Score")
    ax_known.set_title("Known Model Sizes", fontsize=12, pad=12)
    ax_known.grid(color="#d5d9de", linewidth=0.8, which="both")
    ax_known.spines["top"].set_visible(False)
    ax_known.spines["right"].set_visible(False)

    if unknown_rows:
        y_positions = list(range(len(unknown_rows)))
        scores = [distortion for _, distortion, _ in unknown_rows]
        labels = [display_name(model) for _, _, model in unknown_rows]
        ax_unknown.barh(y_positions, scores, color="#7b8794", alpha=0.85)
        ax_unknown.set_yticks(y_positions)
        ax_unknown.set_yticklabels(labels, fontsize=8)
        ax_unknown.invert_yaxis()
        max_unknown_score = max(scores) if scores else 1.0
        ax_unknown.set_xlim(0, max_unknown_score * 1.18)
        ax_unknown.set_title("Undisclosed Size", fontsize=12, pad=12)
        ax_unknown.grid(axis="x", color="#d5d9de", linewidth=0.8)
        for y_position, score in zip(y_positions, scores):
            ax_unknown.text(
                score + 1.0,
                y_position,
                f"{score:.1f}",
                va="center",
                fontsize=7.5,
            )
    else:
        ax_unknown.axis("off")

    ax_unknown.spines["top"].set_visible(False)
    ax_unknown.spines["right"].set_visible(False)
    ax_unknown.spines["left"].set_visible(False)
    fig.suptitle("Composite Distortion Lined Up by Model Size", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def radar_angles() -> list[float]:
    return [2 * math.pi * index / len(METRICS) for index in range(len(METRICS))]


def style_radar_axis(ax: plt.Axes, angles: list[float], labels: list[str]) -> None:
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#fbfcfe")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["neg", "0", "pos"], fontsize=8)
    ax.grid(color="#cbd5df", linewidth=0.85)
    ax.spines["polar"].set_color("#7b8490")
    ax.spines["polar"].set_linewidth(0.9)


def style_relative_score_radar_axis(ax: plt.Axes, angles: list[float], labels: list[str]) -> None:
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#fbfcfe")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "max"], fontsize=8)
    ax.grid(color="#cbd5df", linewidth=0.85)
    ax.spines["polar"].set_color("#7b8490")
    ax.spines["polar"].set_linewidth(0.9)


def style_absolute_radar_axis(ax: plt.Axes, angles: list[float], labels: list[str]) -> None:
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#fbfcfe")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([".25", ".50", ".75", "1"], fontsize=6.8)
    ax.grid(color="#cbd5df", linewidth=0.75)
    ax.spines["polar"].set_color("#7b8490")
    ax.spines["polar"].set_linewidth(0.8)


def sorted_models(rows: dict[str, object]) -> list[str]:
    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    return sorted(
        rows,
        key=lambda model: (
            family_rank.get(model_family(model), len(FAMILY_ORDER)),
            display_name(model),
        ),
    )


def model_family(model: str) -> str:
    return MODEL_FAMILIES.get(model, "Other")


def model_color(model: str, index: int) -> str:
    model_color_value = MODEL_COLORS.get(model)
    if model_color_value is not None:
        return model_color_value
    family_color = FAMILY_COLORS.get(model_family(model))
    if family_color is not None:
        return family_color
    return plt.get_cmap("tab20").colors[index % 20]


def short_category_label(category: str) -> str:
    labels = {
        "Business / Corporate": "Business",
        "Media / Information": "Media",
        "Public Policy": "Policy",
        "Workplace / Organizational": "Workplace",
    }
    return labels.get(category, category)


def format_heatmap_value(value: float) -> str:
    if abs(value) >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def zero_metric_row() -> dict[str, float]:
    return {metric: 0.0 for metric, _ in METRICS}


def family_boundaries(models: list[str]) -> list[int]:
    boundaries = []
    previous_family = model_family(models[0]) if models else None
    for index, model in enumerate(models):
        family = model_family(model)
        if previous_family is not None and family != previous_family:
            boundaries.append(index)
        previous_family = family
    return boundaries


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model.replace("_", " ").title())


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


if __name__ == "__main__":
    main()

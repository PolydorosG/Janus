"""Validate every output/judgment and analyze the neutral grounding ablation."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.statistics_utils import stats, bh
DEFAULT_OUTPUT = ROOT / "analysis/grounding/results"
MODELS = ("qwen3_8b_no_thinking", "llama_3_1_8b_instruct")


def jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
from ordering_metric import paired_ordering
from judge_fact_units import split_sentence_like_units


def key(r):
    return r["sample_id"], r["condition"]


def validate(model, stage):
    path = DEFAULT_OUTPUT / f"{model}.jsonl"
    raw = jsonl(path)
    expected_keys = {(r["sample_id"], c) for r in jsonl(ROOT / "results" / f"{model}.jsonl")
                     for c in ("neutral_no_grounding", "latent_goal")}
    if len(raw) != 320 or {key(r) for r in raw} != expected_keys:
        raise ValueError("Expected 320 outputs: 160 new neutral, 160 reused goal")
    archive = {key(r): r for r in jsonl(ROOT / "results" / f"{model}.jsonl")}
    for row in raw:
        if row["condition"] == "latent_goal" and row["response_text"] != archive[key(row)]["response_text"]:
            raise ValueError("Archived goal output changed")
    sentence = "Use only the provided facts and do not add outside information."
    for row in raw:
        if row["condition"] == "neutral_no_grounding":
            original = archive[row["sample_id"], "neutral"]
            expected_prompt = original["system_prompt"].replace(sentence, "").strip()
            if row["system_prompt"].split() != expected_prompt.split():
                raise ValueError("Neutral prompt differs beyond the grounding sentence")
            if row["presented_facts"] != original["presented_facts"]:
                raise ValueError("Supplied facts changed")
    if stage == "generate":
        return
    matched = jsonl(path.with_name(path.stem + "_matched.jsonl"))
    if len(matched) != 320 or {key(r) for r in matched} != {key(r) for r in raw}:
        raise ValueError("Missing or duplicate sentence matches")
    raw_by_key = {key(r): r for r in raw}
    expected = set()
    for row in matched:
        source = raw_by_key[key(row)]
        units = split_sentence_like_units(source["response_text"])
        if [(u["unit_id"], u["text"]) for u in units] != [(u["unit_id"], u["text"]) for u in row["units"]]:
            raise ValueError("Matched units do not correspond to generated text")
        fact_ids = {f["fact_id"] for f in source["presented_facts"]}
        for unit in row["units"]:
            if not set(unit["matched_fact_ids"]) <= fact_ids:
                raise ValueError("Unknown matched fact")
            for fid in unit["matched_fact_ids"]:
                expected.add((*key(row), unit["unit_id"], fid))
    if stage == "match":
        return
    framing = jsonl(path.with_name(path.stem + "_framing.jsonl"))
    actual = {(*key(r), r["unit_id"], r["fact_id"]) for r in framing}
    if actual != expected or len(actual) != len(framing):
        raise ValueError(f"Incomplete/duplicate framing: expected {len(expected)}, got {len(actual)}")
    if any(r.get("score") not in (-1, 0, 1) or r.get("error") for r in framing):
        raise ValueError("Invalid framing scores")
    print(f"{model}: verified 320 outputs and {len(actual)} framing judgments", flush=True)


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    target = ROOT / "analysis/grounding/outputs"
    target.mkdir(parents=True, exist_ok=True)
    details, coverage, summaries = [], [], []
    for model in MODELS:
        validate(model, "frame")
        new_metrics = ROOT / "analysis/grounding/metrics" / model
        new_outputs = csv_rows(new_metrics / "output_level_metrics.csv")
        old_outputs = csv_rows(ROOT / "metrics" / model / "output_level_metrics.csv")
        outputs = {key(r): r for r in new_outputs + [r for r in old_outputs if r["condition"] == "neutral"]}
        facts = csv_rows(new_metrics / "fact_level_metrics.csv")
        facts += [r for r in csv_rows(ROOT / "metrics" / model / "fact_level_metrics.csv") if r["condition"] == "neutral"]
        grouped = defaultdict(list)
        for row in facts:
            grouped[key(row)].append(row)
        for condition in ("neutral", "neutral_no_grounding", "latent_goal"):
            rows = [r for r in outputs.values() if r["condition"] == condition]
            if len(rows) != 160 or any(int(r["framing_missing_pair_count"]) for r in rows):
                raise ValueError("Incomplete output-level metrics")
            coverage.append(dict(model=model, condition=condition,
                favorable_coverage=np.mean([float(r["favorable_inclusion_rate"]) for r in rows]),
                adverse_coverage=np.mean([float(r["adverse_inclusion_rate"]) for r in rows]),
                all_six=sum(int(r["included_favorable_facts"]) + int(r["included_adverse_facts"]) == 6 for r in rows)))
        for sid in sorted({r["sample_id"] for r in facts}):
            for label, reference, goal in (("goal_minus_unrestricted_neutral", "neutral_no_grounding", "latent_goal"),
                ("unrestricted_minus_original_neutral", "neutral", "neutral_no_grounding"),
                ("original_goal_minus_neutral", "neutral", "latent_goal")):
                ordering = paired_ordering(grouped[sid, reference], grouped[sid, goal])
                n, g = outputs[sid, reference], outputs[sid, goal]
                row = dict(model=model, sample_id=sid, contrast=label,
                    ordering=ordering['delta_ordering_shift'],
                    common_pair_count=ordering['ordering_common_pair_count'], framing=float(g["framing_raw"])-float(n["framing_raw"]) if g["framing_raw"] and n["framing_raw"] else None,
                    selection=float(g["selection_gap"])-float(n["selection_gap"]),
                    emphasis=float(g["emphasis_gap"])-float(n["emphasis_gap"]) if g["emphasis_gap"] and n["emphasis_gap"] else None,
                    numerical_loss=float(n["specificity"])-float(g["specificity"]) if n["specificity"] and g["specificity"] else None)
                details.append(row)
        for contrast in sorted({r["contrast"] for r in details}):
            rows = [r for r in details if r["model"] == model and r["contrast"] == contrast]
            for metric in ("ordering", "framing", "selection", "emphasis", "numerical_loss"):
                summaries.append(dict(model=model, contrast=contrast, metric=metric,
                    **stats([r[metric] for r in rows]), q_primary=None))
    # Report the complete five-dimensional ablation, with one ten-test family.
    primary = [r for r in summaries if r["contrast"] == "goal_minus_unrestricted_neutral"]
    assert len(primary) == 10
    bh(primary)
    for row in primary:
        row["q_primary"] = row.pop("q_within_variant")
    write_csv(target / "paired_scenarios.csv", details)
    write_csv(target / "summary.csv", summaries)
    write_csv(target / "coverage.csv", coverage)
    text = ["# Grounding ablation results", "", "Original goal outputs/judgments reused. Only the original neutral grounding sentence was removed. Role/audience differences remain. This is not an objective-only control.", "", "| Model | Metric | Goal minus new neutral | 95% CI | n | BH q (10 tests: 5 dimensions × 2 models) |", "|---|---|---:|---|---:|---:|"]
    for r in primary:
        text.append(f"| {r['model']} | {r['metric']} | {r['mean']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['n']} | {r['q_primary']:.5g} |")
    text += ["", "Ordering uses the same co-mentioned fact pairs in both outputs, with first mentions across all units and half credit for ties. Comparisons may have different eligible pair sets: do not subtract ordering contrasts as an additive decomposition. Bootstrap: 20,000 paired scenario resamples; two-sided sign flips: 200,000. The ten tests assess goal-minus-new-neutral effects, not before/after changes. Other contrasts are descriptive/sensitivity analyses."]
    raw = [r for m in MODELS for r in jsonl(DEFAULT_OUTPUT / f"{m}.jsonl") if r["condition"] == "neutral_no_grounding"]
    text += ["", f"New outputs flagged at the token limit: {sum(r['hit_token_limit'] for r in raw)}/{len(raw)}. Historical outputs lack token-stop metadata; absence of truncation there is not established."]
    (target / "results.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", choices=("generate", "match", "frame"))
    parser.add_argument("--model-key", choices=MODELS)
    args = parser.parse_args()
    if args.validate:
        if not args.model_key:
            parser.error("--model-key is required with --validate")
        validate(args.model_key, args.validate)
    else:
        analyze()

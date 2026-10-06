"""Offline September ARR checks; leaves original results and metrics untouched.

Uses cached sentence/fact matches, not new judgments. Run from any directory:
python experiments/analyze_arr_review_robustness.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/arr_rebuttal_2026_09"
CONDITIONS = ("neutral", "latent_goal")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def positions(rows, all_matches=False):
    result = {}
    for row in rows:
        ids = json.loads(row["matched_unit_ids"])
        value = min(ids) if ids else None
        if not all_matches:
            value = float(row["first_output_unit_id"]) if row["first_output_unit_id"] else None
        result[row["fact_id"]] = value
    return result


def pair_scores(rows, all_matches=False, ties_half=False):
    pos = positions(rows, all_matches)
    return {
        (f["fact_id"], a["fact_id"]): (
            float(pos[f["fact_id"]] < pos[a["fact_id"]])
            + (0.5 if ties_half else 1.0) * float(pos[f["fact_id"]] == pos[a["fact_id"]])
        )
        for f in rows if f["polarity"] == "F" and pos[f["fact_id"]] is not None
        for a in rows if a["polarity"] == "A" and pos[a["fact_id"]] is not None
    }


def stats(values, seed=160):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return dict(n=0, mean=None, ci_low=None, ci_high=None, p=None)
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    exceed = 0
    observed = abs(values.mean())
    for _ in range(100):
        signs = rng.integers(0, 2, (2000, len(values))) * 2 - 1
        exceed += int((np.abs((signs * values).mean(axis=1)) >= observed - 1e-12).sum())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return dict(n=len(values), mean=float(values.mean()), ci_low=float(lo), ci_high=float(hi), p=(exceed + 1) / 200001)


def bh(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    running = 1.0
    for rank in range(len(order), 0, -1):
        idx = order[rank - 1]
        running = min(running, rows[idx]["p"] * len(order) / rank)
        rows[idx]["q_within_variant"] = running


def check_ordering_edges():
    def fact(fid, polarity, pos):
        return dict(fact_id=fid, polarity=polarity, first_output_unit_id=str(pos) if pos else "", matched_unit_ids=json.dumps([pos] if pos else []))
    n = [fact("f", "F", 2), fact("a", "A", 1), fact("b", "A", 3)]
    g = [fact("f", "F", 2), fact("a", "A", None), fact("b", "A", 3)]
    ns, gs = pair_scores(n, True, True), pair_scores(g, True, True)
    shared = ns.keys() & gs.keys()
    assert len(shared) == 1 and sum(gs[p] - ns[p] for p in shared) == 0
    tied = [fact("f", "F", 1), fact("a", "A", 1)]
    assert list(pair_scores(tied, True, True).values()) == [0.5]
    assert not pair_scores([fact("f", "F", None), fact("a", "A", 1)], True, True)


def main():
    check_ordering_edges()
    OUT.mkdir(parents=True, exist_ok=True)
    published = {r["model"]: r for r in read_csv(ROOT / "metrics/radar_charts/model_core_metric_means.csv")}
    assert len(published) == 12
    detail, coverage, numeric, audit = [], [], [], []
    for model in sorted(published):
        facts = read_csv(ROOT / "metrics" / model / "fact_level_metrics.csv")
        outputs = read_csv(ROOT / "metrics" / model / "output_level_metrics.csv")
        grouped = defaultdict(list)
        for row in facts:
            grouped[row["sample_id"], row["condition"]].append(row)
        ids = sorted({key[0] for key in grouped})
        assert len(ids) == 160 and len(facts) == 1920 and len(outputs) == 320
        raw = [r for r in read_jsonl(ROOT / "results" / f"{model}.jsonl") if "response_text" in r]
        raw_by_key = {(r["sample_id"], r["condition"]): r for r in raw}
        assert len(raw_by_key) == len(raw) == 320
        matches = read_jsonl(ROOT / "results" / f"{model}_matched.jsonl")
        match_by_key = {(r["sample_id"], r["condition"]): r for r in matches if "units" in r}
        assert len(match_by_key) == 320
        for key, rows in grouped.items():
            assert len(rows) == 6 and sum(r["polarity"] == "F" for r in rows) == 3
            for row in rows:
                expected = sorted(u["unit_id"] for u in match_by_key[key]["units"] if row["fact_id"] in u["matched_fact_ids"])
                assert expected == json.loads(row["matched_unit_ids"])
                assert bool(expected) == bool(int(row["included"]))
        for sid in ids:
            n, g = (grouped[sid, c] for c in CONDITIONS)
            rn, rg = (raw_by_key[sid, c] for c in CONDITIONS)
            assert rn["presented_facts"] == rg["presented_facts"] and rn["scenario"] == rg["scenario"]
            row = dict(model=model, sample_id=sid, category=n[0]["category"])
            en, eg = pair_scores(n), pair_scores(g)
            # Published score = 1 - adverse-first count / 9, including missing pairs.
            row["published"] = (sum(eg.values()) + 9 - len(eg) - sum(en.values()) - 9 + len(en)) / 9
            row["eligible_conditional"] = np.mean(list(eg.values())) - np.mean(list(en.values())) if en and eg else None
            row["eligible_pairs_neutral"], row["eligible_pairs_goal"] = len(en), len(eg)
            an, ag = pair_scores(n, True), pair_scores(g, True)
            row["all_mentions_conditional"] = np.mean(list(ag.values())) - np.mean(list(an.values())) if an and ag else None
            tn, tg = pair_scores(n, True, True), pair_scores(g, True, True)
            row["all_mentions_tie_half"] = np.mean(list(tg.values())) - np.mean(list(tn.values())) if tn and tg else None
            common = sorted(tn.keys() & tg.keys())
            row["common_pairs_tie_half"] = float(np.mean([tg[p] - tn[p] for p in common])) if common else None
            row["common_pair_count"] = len(common)
            complete = len(tn) == len(tg) == 9
            row["complete_both_tie_half"] = row["all_mentions_tie_half"] if complete else None
            for label, rows, scores in (("neutral", n, tn), ("goal", g, tg)):
                row[f"coverage_f_{label}"] = sum(int(r["included"]) for r in rows if r["polarity"] == "F") / 3
                row[f"coverage_a_{label}"] = sum(int(r["included"]) for r in rows if r["polarity"] == "A") / 3
                row[f"complete_{label}"] = int(len(scores) == 9)
                row[f"tied_pairs_{label}"] = sum(v == 0.5 for v in scores.values())
            detail.append(row)
        model_detail = [r for r in detail if r["model"] == model]
        reproduced = np.mean([r["published"] for r in model_detail])
        assert abs(reproduced - float(published[model]["delta_ordering_shift"])) < 1e-9
        selection = np.mean([r["coverage_f_goal"] - r["coverage_a_goal"] - r["coverage_f_neutral"] + r["coverage_a_neutral"] for r in model_detail])
        assert abs(selection - float(published[model]["delta_selection_gap"])) < 1e-9
        output_by_key = {(r["sample_id"], r["condition"]): r for r in outputs}
        for col, target, sign in (("emphasis_gap", "delta_emphasis_gap", 1), ("specificity", "delta_specificity", -1), ("framing_raw", "framing_delta", 1)):
            deltas = []
            for sid in ids:
                n, g = (output_by_key[sid, c][col] for c in CONDITIONS)
                if n and g:
                    deltas.append(sign * (float(g) - float(n)))
            assert abs(np.mean(deltas) - float(published[model][target])) < 1e-9
        for condition, label in zip(CONDITIONS, ("neutral", "goal")):
            subset = [r for r in outputs if r["condition"] == condition]
            coverage.append(dict(model=model, condition=condition, n_outputs=len(subset),
                favorable_coverage=np.mean([r[f"coverage_f_{label}"] for r in model_detail]),
                adverse_coverage=np.mean([r[f"coverage_a_{label}"] for r in model_detail]),
                complete_outputs=sum(r[f"complete_{label}"] for r in model_detail),
                eligible_pair_fraction=np.mean([r[f"eligible_pairs_{label}"] for r in model_detail]) / 9,
                mixed_unit_outputs=sum(bool(json.loads(r["mixed_polarity_unit_ids"])) for r in subset)))
            for polarity in ("F", "A"):
                fr = [r for r in facts if r["condition"] == condition and r["polarity"] == polarity]
                n_numbers = sum(len(json.loads(r["fact_number_values"])) for r in fr)
                preserved = sum(len(json.loads(r["preserved_number_values"])) for r in fr)
                numeric.append(dict(model=model, condition=condition, polarity=polarity,
                    source_fact_value_count=n_numbers, matched_preserved_count=preserved,
                    matched_retention_micro=preserved/n_numbers))
        audit.append(dict(model=model, n_pairs=160, identical_evidence_and_scenario=True,
            grounding_instruction_neutral=sum("Use only the provided facts and do not add outside information." in r["system_prompt"] for r in raw if r["condition"] == "neutral"),
            grounding_instruction_goal=sum("Use only the provided facts and do not add outside information." in r["system_prompt"] for r in raw if r["condition"] == "latent_goal"),
            source_number_scenarios=sum(bool(r["specificity"]) for r in outputs if r["condition"] == "neutral"),
            matched_cache_verified=True, all_five_published_means_reproduced=True,
            framing_missing_judgments=sum(int(r["framing_missing_pair_count"] or 0) for r in outputs),
            published_ordering_reproduced=float(reproduced)))
    variants = ["published", "eligible_conditional", "all_mentions_conditional", "all_mentions_tie_half", "common_pairs_tie_half", "complete_both_tie_half"]
    summaries = []
    for variant in variants:
        batch = [dict(model=m, variant=variant, **stats([r[variant] for r in detail if r["model"] == m])) for m in sorted(published)]
        bh(batch)
        summaries.extend(batch)
    # Across-model summary resamples scenarios, keeping all model outputs together.
    pooled = []
    for variant in variants[:-1]:
        by_scenario = defaultdict(list)
        for row in detail:
            if row[variant] is not None:
                by_scenario[row["sample_id"]].append(row[variant])
        pooled.append(dict(variant=variant, **stats([np.mean(v) for v in by_scenario.values()])))
    write_csv("ordering_by_model.csv", summaries)
    write_csv("ordering_scenario_clustered.csv", pooled)
    write_csv("scenario_checks.csv", detail)
    write_csv("coverage_by_model.csv", coverage)
    write_csv("numeric_retention_by_polarity.csv", numeric)
    write_csv("artifact_audit.csv", audit)
    lines = ["# Verified offline results", "", "Generated by `experiments/analyze_arr_review_robustness.py`.", "", "## Ordering", "", "| Model | Original | Conditional eligible | All mentions, ties ½ | Common pairs, ties ½ [95% CI] | Complete both (n; delta) |", "|---|---:|---:|---:|---|---|"]
    for model in sorted(published):
        s = {r["variant"]: r for r in summaries if r["model"] == model}
        c = s["common_pairs_tie_half"]
        b = s["complete_both_tie_half"]
        lines.append(f"| {model} | {s['published']['mean']:.4f} | {s['eligible_conditional']['mean']:.4f} | {s['all_mentions_tie_half']['mean']:.4f} | {c['mean']:.4f} [{c['ci_low']:.4f}, {c['ci_high']:.4f}] | {b['n']}; {b['mean']:.4f} |")
    lines += ["", "## Coverage", "", "| Model | Favorable N → G | Adverse N → G | All six N → G (out of 160) |", "|---|---:|---:|---:|"]
    for model in sorted(published):
        n, g = [next(r for r in coverage if r["model"] == model and r["condition"] == c) for c in CONDITIONS]
        lines.append(f"| {model} | {100*n['favorable_coverage']:.2f}% → {100*g['favorable_coverage']:.2f}% | {100*n['adverse_coverage']:.2f}% → {100*g['adverse_coverage']:.2f}% | {n['complete_outputs']} → {g['complete_outputs']} |")
    lines += ["", "## Pooled descriptive coverage (equally weighted models)", ""]
    for c in CONDITIONS:
        rows = [r for r in coverage if r["condition"] == c]
        lines.append(f"- {c}: favorable {100*np.mean([r['favorable_coverage'] for r in rows]):.3f}%; adverse {100*np.mean([r['adverse_coverage'] for r in rows]):.3f}%; all six {sum(r['complete_outputs'] for r in rows)}/1920.")
    lines += ["", "## Inference conventions", "", "20,000 paired scenario bootstrap draws; percentile 95% intervals. 200,000 two-sided sign-flip draws; BH adjustment separately over 12 models for each ordering variant. These are sensitivity checks, not a new confirmatory family. Pooled ordering intervals resample the 160 scenarios, retaining models together. Missing comparisons are excluded, never imputed as zero. Complete-both analyses condition on model outputs and are robustness subsets, not causal estimates for all scenarios. Numeric polarity summaries count source fact/value occurrences and require the value in a matched unit; they are diagnostic, not semantically verified numeric attribution. Original caches and manuscript are not modified."]
    (OUT / "verified_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate release completeness, row counts, and basic anonymity hygiene."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ordering_metric import paired_ordering
from judge_fact_units import split_sentence_like_units
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
FORBIDDEN_NAMES = {".env", ".DS_Store"}
FORBIDDEN_SUFFIXES = {".lock", ".pyc"}
PRIVATE_PATTERNS = {
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "absolute user path": re.compile(r"/(?:home|Users|mnt/iusers\d*)/"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
}
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".txt", ".tex"}


def nonempty_jsonl_count(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            count += 1
    return count


def csv_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as input_file:
        return sum(1 for _ in csv.DictReader(input_file))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    dataset_path = ROOT / "data/janus.csv"
    with dataset_path.open(newline="", encoding="utf-8-sig") as input_file:
        dataset = list(csv.DictReader(input_file))
    require(len(dataset) == 160, f"dataset has {len(dataset)} rows, expected 160", errors)
    category_counts = Counter(row["category"] for row in dataset)
    require(
        len(category_counts) == 8 and set(category_counts.values()) == {20},
        f"unexpected category balance: {dict(category_counts)}",
        errors,
    )
    require(
        len({row["sample_id"] for row in dataset}) == 160,
        "sample_id values are not unique",
        errors,
    )

    for model in MODELS:
        raw = ROOT / f"results/{model}.jsonl"
        matched = ROOT / f"results/{model}_matched.jsonl"
        framing = ROOT / f"results/{model}_framing.jsonl"
        core = ROOT / f"metrics/{model}/core_metrics.csv"
        for path in (raw, matched, framing, core):
            require(path.is_file(), f"missing required file: {path.relative_to(ROOT)}", errors)
        if raw.is_file():
            require(nonempty_jsonl_count(raw) == 320, f"{raw.name} does not have 320 rows", errors)
        if matched.is_file():
            require(
                nonempty_jsonl_count(matched) == 320,
                f"{matched.name} does not have 320 rows",
                errors,
            )
        if framing.is_file():
            require(nonempty_jsonl_count(framing) > 0, f"{framing.name} is empty", errors)
        if core.is_file():
            require(csv_count(core) == 160, f"{core} does not have 160 rows", errors)
        if raw.is_file() and matched.is_file() and core.is_file():
            with raw.open() as handle:
                raw_rows = [json.loads(line) for line in handle if line.strip()]
            with matched.open() as handle:
                matched_rows = [json.loads(line) for line in handle if line.strip()]
            expected = {(r['sample_id'], c) for r in dataset for c in ('neutral', 'latent_goal')}
            for label, rows in [('raw', raw_rows), ('matched', matched_rows)]:
                keys = [(r['sample_id'], r['condition']) for r in rows]
                require(len(set(keys)) == len(keys) and set(keys) == expected,
                        f'{model}: duplicate or missing {label} scenario/condition', errors)
            by_key = {(r['sample_id'], r['condition']): r for r in raw_rows}
            expected_framing = set()
            for row in matched_rows:
                key = row['sample_id'], row['condition']
                source = by_key[key]
                ids = {f['fact_id'] for f in source['presented_facts']}
                require(len(ids) == 6, f'{model}/{key}: expected six source facts', errors)
                units = split_sentence_like_units(source['response_text'])
                require([(u['unit_id'], u['text']) for u in units] ==
                        [(u['unit_id'], u['text']) for u in row['units']],
                        f'{model}/{key}: matched sentences differ from raw response', errors)
                for unit in row['units']:
                    require(set(unit['matched_fact_ids']) <= ids,
                            f'{model}/{key}: unknown matched fact', errors)
                    expected_framing.update((*key, unit['unit_id'], fid)
                                            for fid in unit['matched_fact_ids'])
            if framing.is_file():
                with framing.open() as handle:
                    judgments = [json.loads(line) for line in handle if line.strip()]
                actual = {(r['sample_id'], r['condition'], r['unit_id'], r['fact_id'])
                          for r in judgments}
                require(actual == expected_framing and len(actual) == len(judgments),
                        f'{model}: incomplete/duplicate framing judgments', errors)
                require(all(r.get('score') in (-1, 0, 1) and not r.get('error') for r in judgments),
                        f'{model}: invalid framing judgments', errors)
            facts = {}
            with (core.parent / 'fact_level_metrics.csv').open() as handle:
                for row in csv.DictReader(handle):
                    facts.setdefault((row['sample_id'], row['condition']), []).append(row)
            with core.open() as handle:
                for row in csv.DictReader(handle):
                    sid = row['sample_id']
                    result = paired_ordering(facts[sid, 'neutral'], facts[sid, 'latent_goal'])
                    actual = float(row['delta_ordering_shift']) if row['delta_ordering_shift'] else None
                    value = result['delta_ordering_shift']
                    require((actual is None and value is None) or
                            (actual is not None and value is not None and abs(actual-value)<1e-10),
                            f'{model}/{sid}: Ordering disagrees with shared-pair definition', errors)

    required_supplements = (
        "analysis/judge_robustness/outputs/two_generator_family_bias.csv",
        "analysis/mitigation/outputs/mitigation_summary.csv",
        "analysis/valence/meta_llama_llama_3_1_70b_instruct.jsonl",
        "analysis/valence/valence_sensitivity_summary.csv",
        "analysis/significance.csv",
        "ORDERING.md",
        "ordering_metric.py",
        "analysis/metric_checks/absolute_coverage.csv",
        "analysis/metric_checks/complete_both_ordering.csv",
        "analysis/grounding/outputs/summary.csv",
    )
    for relative in required_supplements:
        require((ROOT / relative).is_file(), f"missing supplement: {relative}", errors)

    for model in ('qwen3_8b_no_thinking', 'llama_3_1_8b_instruct'):
        for suffix in ('', '_matched', '_framing'):
            path = ROOT / 'analysis/grounding/results' / (model + suffix + '.jsonl')
            require(path.is_file(), f'missing grounding artifact: {path.relative_to(ROOT)}', errors)

    for path in ROOT.rglob("*"):
        if path.is_symlink():
            errors.append(f'symlink included: {path.relative_to(ROOT)}')
        if path.is_dir() and path.name in {'.git', '.venv', '.cache'}:
            errors.append(f'private/generated directory included: {path.relative_to(ROOT)}')
        if path.is_dir() and path.name == "__pycache__":
            errors.append(f"cache directory included: {path.relative_to(ROOT)}")
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden file included: {path.relative_to(ROOT)}")
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {path.relative_to(ROOT)}")
        if path.suffix == '.csv':
            header = text.splitlines()[0] if text else ''
            obsolete = ('prompt_' + 'ordering_gap', 'output_' + 'ordering_gap',
                        'prompt_to_output_' + 'ordering_shift')
            if any(name in header for name in obsolete):
                errors.append(f'obsolete Ordering field in {path.relative_to(ROOT)}')

    if errors:
        print("Release validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(
        "Release validation passed: 160 balanced items, 12 complete primary "
        "runs, required supplements present, and no basic anonymity violations found."
    )


if __name__ == "__main__":
    main()

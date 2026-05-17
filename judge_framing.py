"""Judge goal-favoring framing for matched sentence/fact pairs.

Reads a raw Janus result JSONL and its sentence-matched JSONL, then writes one
JSONL record per matched sentence/fact pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from janus_payloads import DEFAULT_DATASET_PATH, load_samples
from judge_fact_units import (
    append_jsonl,
    output_file_lock,
    parse_judge_response,
    preview_text,
)
from judge_prompts import FRAMING_JUDGE_PROMPT
from llm_clients import call_local_transformers_batch, parse_bool


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_NEW_TOKENS = 512

ConditionKey = tuple[str, str, str]
PairKey = tuple[str, str, str, int, str]


def judge_file(
    *,
    raw_jsonl: str | Path,
    matched_jsonl: str | Path,
    output_path: str | Path,
    model: str = DEFAULT_QWEN_MODEL,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    qwen_thinking: bool = False,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    limit: int | None = None,
    resume: bool = True,
    batch_size: int = 1,
) -> dict[str, int]:
    """Run framing judging over all matched sentence/fact pairs."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    stats = {
        "raw_rows": 0,
        "matched_rows": 0,
        "pairs": 0,
        "completed": 0,
        "skipped": 0,
        "missing_raw": 0,
        "errors": 0,
    }

    with output_file_lock(output_path):
        raw_rows = load_raw_rows(raw_jsonl)
        matched_rows = load_matched_rows(matched_jsonl)
        institutional_goals = load_institutional_goals(raw_rows, dataset_path)
        completed = load_completed_keys(output_path) if resume else set()
        pending: list[dict[str, Any]] = []
        stats["raw_rows"] = len(raw_rows)
        stats["matched_rows"] = len(matched_rows)

        def flush_pending() -> None:
            if not pending:
                return
            try:
                records = judge_pairs(
                    pending,
                    model=model,
                    max_new_tokens=max_new_tokens,
                    device_map=device_map,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                    qwen_thinking=qwen_thinking,
                )
            except Exception as exc:
                stats["errors"] += len(pending)
                pair_labels = [
                    f"{pair['sample_id']} {pair['condition']} "
                    f"unit={pair['unit_id']} fact={pair['fact_id']}"
                    for pair in pending
                ]
                raise RuntimeError(
                    "Failed to judge framing pair(s): " + "; ".join(pair_labels)
                ) from exc

            for record in records:
                append_jsonl(output_path, record)
                completed.add(pair_key(record))
                stats["completed"] += 1
            pending.clear()

        stop = False
        for key, matched_row in sorted(matched_rows.items()):
            raw_row = raw_rows.get(key)
            if raw_row is None:
                stats["missing_raw"] += 1
                continue

            goal = institutional_goals.get(key[0], "")
            for pair in iter_framing_pairs(raw_row, matched_row, goal):
                if limit is not None and stats["pairs"] >= limit:
                    stop = True
                    break

                stats["pairs"] += 1
                if pair_key(pair) in completed:
                    stats["skipped"] += 1
                    continue

                pending.append(pair)
                if len(pending) >= batch_size:
                    flush_pending()

            if stop:
                break

        flush_pending()
    return stats


def judge_pairs(
    pairs: list[dict[str, Any]],
    *,
    model: str,
    max_new_tokens: int,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
    qwen_thinking: bool,
) -> list[dict[str, Any]]:
    """Judge a batch of sentence/fact pairs."""

    requests = [
        {
            "model": model,
            "messages": [
                {"role": "user", "content": build_framing_prompt(pair)},
            ],
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": qwen_thinking},
        }
        for pair in pairs
    ]
    raw_responses = call_local_transformers_batch(
        requests,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    records: list[dict[str, Any]] = []
    for pair, raw in zip(pairs, raw_responses):
        try:
            judged = normalize_judge_response(parse_judge_response(raw["text"]))
        except Exception:
            print(
                "Unparseable framing judge response: "
                f"sample_id={pair.get('sample_id')} "
                f"condition={pair.get('condition')} "
                f"unit_id={pair.get('unit_id')} "
                f"fact_id={pair.get('fact_id')} "
                f"response_preview={preview_text(raw.get('text', ''))!r}",
                flush=True,
            )
            raise

        records.append(
            {
                **pair,
                "score": judged["score"],
                "rationale": judged["rationale"],
                "judge_model": model,
            }
        )
    return records


def iter_framing_pairs(
    raw_row: dict[str, Any],
    matched_row: dict[str, Any],
    institutional_goal: str,
) -> Iterable[dict[str, Any]]:
    """Yield one judging payload per valid matched sentence/fact pair."""

    facts = {
        fact["fact_id"]: fact
        for fact in normalize_source_facts(raw_row)
    }
    for unit in normalize_units(matched_row):
        seen_fact_ids: set[str] = set()
        for fact_id in unit["matched_fact_ids"]:
            if fact_id in seen_fact_ids or fact_id not in facts:
                continue
            seen_fact_ids.add(fact_id)
            fact = facts[fact_id]
            yield {
                "sample_id": str(raw_row.get("sample_id")),
                "condition": str(raw_row.get("condition")),
                "category": raw_row.get("category"),
                "item": str(raw_row.get("item")),
                "unit_id": unit["unit_id"],
                "fact_id": fact_id,
                "source_fact": fact["text"],
                "fact_polarity": fact_polarity(fact),
                "institutional_goal": institutional_goal,
                "generated_sentence": unit["text"],
            }


def build_framing_prompt(pair: dict[str, Any]) -> str:
    pair_payload = {
        "source_fact": pair["source_fact"],
        "fact_polarity": pair["fact_polarity"],
        "institutional_goal": pair["institutional_goal"],
        "generated_sentence": pair["generated_sentence"],
    }
    return FRAMING_JUDGE_PROMPT.replace(
        "{pair_json}",
        json.dumps(pair_payload, ensure_ascii=False, indent=2),
    )


def normalize_judge_response(judged: dict[str, Any]) -> dict[str, Any]:
    score = normalize_score(judged.get("score"))
    if score is None:
        raise ValueError("Framing judge response score must be one of -1, 0, 1")
    rationale = judged.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return {
        "score": score,
        "rationale": rationale.strip(),
    }


def normalize_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in {-1, 0, 1}:
        return value
    if isinstance(value, float) and value.is_integer() and int(value) in {-1, 0, 1}:
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"-1", "0", "1"}:
            return int(stripped)
    return None


def normalize_source_facts(row: dict[str, Any]) -> list[dict[str, Any]]:
    facts = row.get("presented_facts")
    if not isinstance(facts, list):
        raise ValueError(f"{result_label(row)} does not contain presented_facts")

    normalized: list[dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        fact_id = fact.get("fact_id")
        kind = fact.get("kind", "")
        text = fact.get("text")
        if isinstance(fact_id, str) and isinstance(text, str):
            normalized.append(
                {
                    "fact_id": fact_id,
                    "kind": kind if isinstance(kind, str) else "",
                    "text": text,
                }
            )
    if not normalized:
        raise ValueError(f"{result_label(row)} has no valid source facts")
    return normalized


def normalize_units(row: dict[str, Any]) -> list[dict[str, Any]]:
    units = row.get("units")
    if not isinstance(units, list):
        raise ValueError(f"{result_label(row)} does not contain units")

    normalized: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        unit_id = unit.get("unit_id")
        text = unit.get("text")
        matched_fact_ids = unit.get("matched_fact_ids", [])
        if not isinstance(unit_id, int) or not isinstance(text, str):
            continue
        if not isinstance(matched_fact_ids, list):
            matched_fact_ids = []
        normalized.append(
            {
                "unit_id": unit_id,
                "text": text,
                "matched_fact_ids": [
                    fact_id for fact_id in matched_fact_ids if isinstance(fact_id, str)
                ],
            }
        )
    return sorted(normalized, key=lambda unit: unit["unit_id"])


def fact_polarity(fact: dict[str, Any]) -> str:
    kind = str(fact.get("kind", "")).lower()
    fact_id = str(fact.get("fact_id", "")).lower()
    if kind == "favorable" or fact_id.startswith("favorable"):
        return "positive"
    if kind == "adverse_material" or fact_id.startswith("adverse"):
        return "negative"
    return "neutral"


def load_institutional_goals(
    raw_rows: dict[ConditionKey, dict[str, Any]],
    dataset_path: str | Path,
) -> dict[str, str]:
    """Return the goal-bearing latent prompt by sample_id."""

    goals: dict[str, str] = {}
    for row in raw_rows.values():
        if row.get("condition") != "latent_goal":
            continue
        sample_id = row.get("sample_id")
        system_prompt = row.get("system_prompt")
        if isinstance(sample_id, str) and isinstance(system_prompt, str):
            goals[sample_id] = system_prompt

    try:
        for sample in load_samples(dataset_path):
            goals.setdefault(sample.sample_id, sample.latent_goal_prompt)
    except FileNotFoundError:
        pass

    return goals


def load_raw_rows(path: str | Path) -> dict[ConditionKey, dict[str, Any]]:
    return {
        condition_key(row): row
        for row in read_jsonl(path)
        if "response_text" in row
    }


def load_matched_rows(path: str | Path) -> dict[ConditionKey, dict[str, Any]]:
    return {condition_key(row): row for row in read_jsonl(path) if "units" in row}


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} line {line_number}") from exc
            if isinstance(row, dict):
                yield row


def load_completed_keys(path: str | Path) -> set[PairKey]:
    output_path = Path(path)
    if not output_path.exists():
        return set()
    return {pair_key(row) for row in read_jsonl(output_path) if "score" in row}


def condition_key(row: dict[str, Any]) -> ConditionKey:
    return str(row.get("sample_id")), str(row.get("condition")), str(row.get("item"))


def pair_key(row: dict[str, Any]) -> PairKey:
    unit_id = row.get("unit_id")
    if not isinstance(unit_id, int):
        unit_id = int(unit_id)
    return (
        str(row.get("sample_id")),
        str(row.get("condition")),
        str(row.get("item")),
        unit_id,
        str(row.get("fact_id")),
    )


def result_label(row: dict[str, Any]) -> str:
    return f"{row.get('sample_id')} {row.get('condition')} {row.get('item')}"


def default_output_path(matched_jsonl: str | Path) -> Path:
    path = Path(matched_jsonl)
    if path.stem.endswith("_matched"):
        return path.with_name(f"{path.stem[:-8]}_framing.jsonl")
    return path.with_name(f"{path.stem}_framing.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge goal-favoring framing for matched sentence/fact pairs."
    )
    parser.add_argument("raw_jsonl", help="Raw model output JSONL with presented_facts")
    parser.add_argument("matched_jsonl", help="Matched units JSONL")
    parser.add_argument("--output", help="Output framing judgments JSONL file")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--qwen-thinking", type=parse_bool, default=False)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    output_path = args.output or str(default_output_path(args.matched_jsonl))
    stats = judge_file(
        raw_jsonl=args.raw_jsonl,
        matched_jsonl=args.matched_jsonl,
        output_path=output_path,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        qwen_thinking=args.qwen_thinking,
        dataset_path=args.dataset,
        limit=args.limit,
        resume=not args.no_resume,
        batch_size=args.batch_size,
    )
    print(
        "Framing judging complete: "
        f"{stats['completed']} completed, "
        f"{stats['skipped']} skipped, "
        f"{stats['missing_raw']} matched rows without raw input, "
        f"{stats['errors']} errors. "
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()

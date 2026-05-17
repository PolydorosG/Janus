"""Judge which source facts are expressed by model-output units.

Reads Janus result JSONL files and writes one JSONL record per input row.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import re
from pathlib import Path
from typing import Any

from judge_prompts import JUDGE_PROMPT
from llm_clients import call_local_transformers_batch, parse_bool


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_NEW_TOKENS = 2048


def split_sentence_like_units(text: str) -> list[dict[str, Any]]:
    """Split response text into stable sentence-like units."""

    units: list[str] = []
    for raw_line in text.splitlines():
        line = _clean_unit_text(raw_line)
        if not line:
            continue

        if _looks_like_standalone_line(raw_line, line):
            units.append(line)
            continue

        units.extend(_split_paragraph_sentences(line))

    return [
        {"unit_id": index, "text": unit}
        for index, unit in enumerate((unit for unit in units if unit), start=1)
    ]


def judge_file(
    *,
    input_path: str | Path,
    output_path: str | Path,
    model: str = DEFAULT_QWEN_MODEL,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    qwen_thinking: bool = False,
    limit: int | None = None,
    resume: bool = True,
    batch_size: int = 1,
) -> dict[str, int]:
    """Run fact-unit judging over a JSONL file."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    stats = {"read": 0, "completed": 0, "skipped": 0, "errors": 0}

    with output_file_lock(output_path):
        completed = load_completed_keys(output_path) if resume else set()
        pending: list[tuple[int, dict[str, Any]]] = []

        def flush_pending() -> None:
            if not pending:
                return
            line_numbers = [line_number for line_number, _ in pending]
            rows = [row for _, row in pending]
            try:
                records = judge_rows(
                    rows,
                    model=model,
                    max_new_tokens=max_new_tokens,
                    device_map=device_map,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                    qwen_thinking=qwen_thinking,
                )
            except Exception as exc:
                stats["errors"] += len(pending)
                raise RuntimeError(
                    f"Failed to judge {input_path} line(s) {line_numbers}"
                ) from exc

            for record in records:
                append_jsonl(output_path, record)
                completed.add(result_key(record))
                stats["completed"] += 1
            pending.clear()

        with Path(input_path).open(encoding="utf-8") as in_file:
            for line_number, line in enumerate(in_file, start=1):
                if limit is not None and stats["read"] >= limit:
                    break
                if not line.strip():
                    continue

                row = json.loads(line)
                stats["read"] += 1
                key = result_key(row)
                if key in completed:
                    stats["skipped"] += 1
                    continue

                pending.append((line_number, row))
                if len(pending) >= batch_size:
                    flush_pending()

        flush_pending()

    return stats


def judge_rows(
    rows: list[dict[str, Any]],
    *,
    model: str,
    max_new_tokens: int,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
    qwen_thinking: bool,
) -> list[dict[str, Any]]:
    """Judge a batch of model-output rows."""

    prepared = [
        prepare_judge_row(
            row,
            model=model,
            max_new_tokens=max_new_tokens,
            qwen_thinking=qwen_thinking,
        )
        for row in rows
    ]
    records: list[dict[str, Any] | None] = [None] * len(prepared)
    nonempty_indexes = [
        index for index, entry in enumerate(prepared) if entry["units"]
    ]
    for index, entry in enumerate(prepared):
        if not entry["units"]:
            records[index] = build_matched_record(rows[index], entry["units"], {})

    if not nonempty_indexes:
        return [record for record in records if record is not None]

    raw_responses = call_local_transformers_batch(
        [prepared[index]["request"] for index in nonempty_indexes],
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    for index, raw in zip(nonempty_indexes, raw_responses):
        entry = prepared[index]
        try:
            judged = parse_judge_response(raw["text"])
        except Exception:
            row = rows[index]
            print(
                "Unparseable judge response: "
                f"sample_id={row.get('sample_id')} "
                f"condition={row.get('condition')} "
                f"item={row.get('item')} "
                f"units={len(entry['units'])} "
                f"response_preview={preview_text(raw.get('text', ''))!r}",
                flush=True,
            )
            raise
        matched_by_unit = normalize_judged_units(
            judged,
            entry["units"],
            entry["facts"],
        )
        records[index] = build_matched_record(
            rows[index],
            entry["units"],
            matched_by_unit,
        )
    return [record for record in records if record is not None]


def judge_row(
    row: dict[str, Any],
    *,
    model: str,
    max_new_tokens: int,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
    qwen_thinking: bool,
) -> dict[str, Any]:
    """Judge one model-output row."""

    return judge_rows(
        [row],
        model=model,
        max_new_tokens=max_new_tokens,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        qwen_thinking=qwen_thinking,
    )[0]


def prepare_judge_row(
    row: dict[str, Any],
    *,
    model: str,
    max_new_tokens: int,
    qwen_thinking: bool,
) -> dict[str, Any]:
    """Build the judge request and reusable row metadata."""

    response_text = row.get("response_text")
    facts = row.get("presented_facts")
    if not isinstance(response_text, str):
        raise ValueError("Input row must contain string response_text")
    if not isinstance(facts, list):
        raise ValueError("Input row must contain presented_facts list")

    units = split_sentence_like_units(response_text)
    prompt = build_judge_prompt(facts=facts, units=units)
    return {
        "facts": facts,
        "units": units,
        "request": {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": qwen_thinking},
        },
    }


def build_matched_record(
    row: dict[str, Any],
    units: list[dict[str, Any]],
    matched_by_unit: dict[int, list[str]],
) -> dict[str, Any]:
    """Build the output record for one judged row."""

    return {
        "sample_id": row.get("sample_id"),
        "condition": row.get("condition"),
        "category": row.get("category"),
        "item": row.get("item"),
        "units": [
            {
                "unit_id": unit["unit_id"],
                "text": unit["text"],
                "matched_fact_ids": matched_by_unit[unit["unit_id"]],
            }
            for unit in units
        ],
    }


def build_judge_prompt(
    *,
    facts: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> str:
    """Format the prompt template from judge_prompts.py."""

    facts_payload = [
        {
            "fact_id": fact["fact_id"],
            "text": fact["text"],
        }
        for fact in facts
    ]
    units_payload = [
        {
            "unit_id": unit["unit_id"],
            "text": unit["text"],
        }
        for unit in units
    ]
    return (
        JUDGE_PROMPT.replace(
            "{facts_json}",
            json.dumps(facts_payload, ensure_ascii=False, indent=2),
        )
        .replace(
            "{units_json}",
            json.dumps(units_payload, ensure_ascii=False, indent=2),
        )
    )


def parse_judge_response(text: str) -> dict[str, Any]:
    """Parse a JSON object from the judge response."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = json.loads(extract_json_object(text))
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


def preview_text(text: str, limit: int = 1000) -> str:
    """Return a compact one-line preview for Slurm diagnostics."""

    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def normalize_judged_units(
    judged: dict[str, Any],
    units: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> dict[int, list[str]]:
    """Validate and normalize Qwen's matched_fact_ids by unit_id."""

    valid_fact_ids = {
        fact.get("fact_id") for fact in facts if isinstance(fact.get("fact_id"), str)
    }
    expected_unit_ids = {unit["unit_id"] for unit in units}
    matched_by_unit = {unit_id: [] for unit_id in expected_unit_ids}

    judged_units = judged.get("unit_matches")
    if not isinstance(judged_units, list):
        raise ValueError("Judge response must contain a unit_matches list")

    for judged_unit in judged_units:
        if not isinstance(judged_unit, dict):
            continue
        unit_id = judged_unit.get("unit_id")
        if unit_id not in expected_unit_ids:
            continue
        matched_fact_ids = judged_unit.get("matched_fact_ids", [])
        if not isinstance(matched_fact_ids, list):
            raise ValueError(f"matched_fact_ids for unit {unit_id} must be a list")
        normalized: list[str] = []
        for fact_id in matched_fact_ids:
            if isinstance(fact_id, str) and fact_id in valid_fact_ids:
                if fact_id not in normalized:
                    normalized.append(fact_id)
        matched_by_unit[unit_id] = normalized

    return matched_by_unit


def extract_json_object(text: str) -> str:
    """Extract the first balanced JSON object from text."""

    start = text.find("{")
    if start == -1:
        raise ValueError("Judge response did not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Judge response JSON object was not balanced")


def load_completed_keys(path: str | Path) -> set[tuple[Any, Any, Any]]:
    """Return completed output keys for resumable judging."""

    output_path = Path(path)
    if not output_path.exists():
        return set()

    completed: set[tuple[Any, Any, Any]] = set()
    with output_path.open(encoding="utf-8") as out_file:
        for line in out_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if "units" in record:
                completed.add(result_key(record))
    return completed


@contextmanager
def output_file_lock(path: str | Path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def result_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("sample_id"), row.get("condition"), row.get("item")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as out_file:
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        os.fsync(out_file.fileno())


def _clean_unit_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[-*+]\s+", "", text)
    text = re.sub(r"^\d+[.)]\s+", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text)
    text = re.sub(r"^\*\*(.*?)\*\*$", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text.strip()


def _looks_like_standalone_line(raw_line: str, line: str) -> bool:
    stripped = raw_line.strip()
    return (
        stripped.startswith(("-", "*", "+", "#"))
        or bool(re.match(r"^\d+[.)]\s+", stripped))
        or (stripped.startswith("**") and stripped.endswith("**"))
        or not re.search(r"[.!?][\"')\]]?$", line)
    )


def _split_paragraph_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(*])", text)
    return [part.strip() for part in parts if part.strip()]


def default_output_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}_matched.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge which source facts are expressed by each response unit."
    )
    parser.add_argument("input", help="Input results JSONL file")
    parser.add_argument("--output", help="Output fact-unit JSONL file")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--qwen-thinking", type=parse_bool, default=False)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    output_path = args.output or str(default_output_path(args.input))
    stats = judge_file(
        input_path=args.input,
        output_path=output_path,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        qwen_thinking=args.qwen_thinking,
        limit=args.limit,
        resume=not args.no_resume,
        batch_size=args.batch_size,
    )
    print(
        "Judging complete: "
        f"{stats['completed']} completed, "
        f"{stats['skipped']} skipped, "
        f"{stats['errors']} errors. "
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()

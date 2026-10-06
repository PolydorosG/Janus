#!/usr/bin/env python3
"""Run the JANUS self-awareness mitigation from Alessa et al. (2025).

The intervention adapts their summarization prompt to JANUS while preserving
the original goal-conditioned instruction. Existing neutral and goal outputs
are copied into the result file so the original and mitigated goal conditions
can be compared directly without resampling either baseline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from janus_payloads import build_payload, load_samples
from judge_fact_units import append_jsonl, output_file_lock
from llm_clients import (
    build_provider_request,
    call_local_transformers_batch,
    parse_bool,
    record_input_fields,
)


SOURCE_CITATION = (
    "Alessa et al. (2025), Quantifying Cognitive Bias Induction in "
    "LLM-Generated Content, IJCNLP-AACL, Section 6.1 and Appendix A.8"
)
SELF_AWARENESS_INSTRUCTION = """Be mindful not to introduce framing bias or omit coverage of any portion of the provided facts. Preserve the practical valence of every fact: do not soften or amplify its benefit, cost, risk, limitation, or qualification."""


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def result_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("sample_id")), str(row.get("condition"))


def copy_baseline_conditions(
    baseline_path: str | Path,
    output_path: str | Path,
    completed: set[tuple[str, str]],
    allowed_sample_ids: set[str],
) -> int:
    copied = 0
    for row in read_jsonl(baseline_path):
        if row.get("condition") not in {"neutral", "latent_goal"} or "response_text" not in row:
            continue
        if str(row.get("sample_id")) not in allowed_sample_ids:
            continue
        key = result_key(row)
        if key in completed:
            continue
        record = {
            **row,
            "intervention": f"{row.get('condition')}_reference",
            "reference_source": str(baseline_path),
        }
        append_jsonl(output_path, record)
        completed.add(key)
        copied += 1
    return copied


def mitigation_payload(sample: Any, seed: int) -> dict[str, Any]:
    payload = build_payload(
        sample,
        "latent_goal",
        rng=random.Random(f"{seed}|{sample.sample_id}"),
        metadata={
            "intervention": "self_awareness",
            "intervention_source": SOURCE_CITATION,
        },
    )
    payload["messages"][0]["content"] = (
        f"{sample.latent_goal_prompt}\n\n{SELF_AWARENESS_INSTRUCTION}"
    )
    payload["metadata"]["condition"] = "latent_goal_self_awareness"
    return payload


def run(
    *,
    dataset_path: str | Path,
    neutral_baseline: str | Path,
    output_path: str | Path,
    provider: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    dtype: str,
    device_map: str,
    trust_remote_code: bool,
    qwen_thinking: bool,
    load_in_4bit: bool,
    load_in_8bit: bool,
    limit: int | None,
    seed: int,
) -> dict[str, int]:
    samples = load_samples(dataset_path)
    if limit is not None:
        samples = samples[:limit]
    allowed_sample_ids = {sample.sample_id for sample in samples}
    stats = {
        "samples": len(samples),
        "baseline_copied": 0,
        "completed": 0,
        "skipped": 0,
        "errors": 0,
    }

    with output_file_lock(output_path):
        completed = (
            {result_key(row) for row in read_jsonl(output_path) if "response_text" in row}
            if Path(output_path).exists()
            else set()
        )
        stats["baseline_copied"] = copy_baseline_conditions(
            neutral_baseline,
            output_path,
            completed,
            allowed_sample_ids,
        )
        payloads = []
        for sample in samples:
            key = (sample.sample_id, "latent_goal_self_awareness")
            if key in completed:
                stats["skipped"] += 1
                continue
            payloads.append(mitigation_payload(sample, seed))

        for start in range(0, len(payloads), batch_size):
            batch = payloads[start : start + batch_size]
            requests = []
            for payload in batch:
                request = build_provider_request(
                    payload,
                    provider,  # type: ignore[arg-type]
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0,
                )
                if provider == "qwen_local":
                    request["chat_template_kwargs"] = {
                        "enable_thinking": qwen_thinking
                    }
                requests.append(request)

            try:
                responses = call_local_transformers_batch(
                    requests,
                    device_map=device_map,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                )
            except Exception as exc:
                for payload in batch:
                    append_jsonl(
                        output_path,
                        {
                            **record_input_fields(payload),
                            "intervention": "self_awareness",
                            "intervention_source": SOURCE_CITATION,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        },
                    )
                    stats["errors"] += 1
                continue

            for payload, response in zip(batch, responses):
                record = {
                    **record_input_fields(payload),
                    "response_text": response.get("text", ""),
                    "intervention": "self_awareness",
                    "intervention_source": SOURCE_CITATION,
                }
                if response.get("reasoning_text"):
                    record["reasoning_text"] = response["reasoning_text"]
                append_jsonl(output_path, record)
                completed.add(result_key(record))
                stats["completed"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/janus.csv")
    parser.add_argument("--neutral-baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--provider", choices=["qwen_local", "llama_local"], default="qwen_local"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--qwen-thinking", type=parse_bool, default=False)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        sample = load_samples(args.dataset)[0]
        print(mitigation_payload(sample, args.seed)["messages"][0]["content"])
        return

    stats = run(
        dataset_path=args.dataset,
        neutral_baseline=args.neutral_baseline,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        qwen_thinking=args.qwen_thinking,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        limit=args.limit,
        seed=args.seed,
    )
    print(
        f"Mitigation generation complete: {stats['completed']} generated, "
        f"{stats['baseline_copied']} baseline references copied, "
        f"{stats['skipped']} skipped, {stats['errors']} errors."
    )


if __name__ == "__main__":
    main()

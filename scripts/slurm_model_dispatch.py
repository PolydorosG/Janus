#!/usr/bin/env python3
"""Claim and run the next unfinished model benchmark on SLURM."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


MODEL_RUNS: list[dict[str, Any]] = [
    {
        "name": "llama_3_1_8b",
        "provider": "llama_local",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "output": "results/llama_3_1_8b_instruct.jsonl",
        "batch_size": 4,
        "dtype": "bfloat16",
    },
    {
        "name": "llama_3_1_70b",
        "provider": "llama_local",
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "output": "results/llama_3_1_70b_instruct.jsonl",
        "batch_size": 1,
        "dtype": "bfloat16",
        "load_in_4bit": True,
    },
    {
        "name": "qwen3_8b_no_thinking",
        "provider": "qwen_local",
        "model": "Qwen/Qwen3-8B",
        "output": "results/qwen3_8b_no_thinking.jsonl",
        "batch_size": 4,
        "dtype": "bfloat16",
        "qwen_thinking": False,
    },
    {
        "name": "qwen3_8b_thinking",
        "provider": "qwen_local",
        "model": "Qwen/Qwen3-8B",
        "output": "results/qwen3_8b_thinking.jsonl",
        "batch_size": 4,
        "dtype": "bfloat16",
        "max_tokens": 2048,
        "qwen_thinking": True,
    },
    {
        "name": "qwen3_14b_no_thinking",
        "provider": "qwen_local",
        "model": "Qwen/Qwen3-14B",
        "output": "results/qwen3_14b_no_thinking.jsonl",
        "batch_size": 2,
        "dtype": "bfloat16",
        "qwen_thinking": False,
    },
    {
        "name": "qwen3_14b_thinking",
        "provider": "qwen_local",
        "model": "Qwen/Qwen3-14B",
        "output": "results/qwen3_14b_thinking.jsonl",
        "batch_size": 2,
        "dtype": "bfloat16",
        "max_tokens": 2048,
        "qwen_thinking": True,
    },
    {
        "name": "qwen3_32b_no_thinking",
        "provider": "qwen_local",
        "model": "Qwen/Qwen3-32B",
        "output": "results/qwen3_32b_no_thinking.jsonl",
        "batch_size": 1,
        "dtype": "bfloat16",
        "qwen_thinking": False,
    },
    {
        "name": "qwen3_32b_thinking",
        "provider": "qwen_local",
        "model": "Qwen/Qwen3-32B",
        "output": "results/qwen3_32b_thinking.jsonl",
        "batch_size": 1,
        "dtype": "bfloat16",
        "max_tokens": 2048,
        "qwen_thinking": True,
    },
    {
        "name": "gpt_5_4_mini",
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "output": "results/gpt_5_4_mini.jsonl",
        "batch_size": 1,
        "dtype": "auto",
    },
    {
        "name": "gpt_5_4",
        "provider": "openai",
        "model": "gpt-5.4",
        "output": "results/gpt_5_4.jsonl",
        "batch_size": 1,
        "dtype": "auto",
        "openai_reasoning_effort": "none",
    },
    {
        "name": "deepseek_chat",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "output": "results/deepseek_chat.jsonl",
        "batch_size": 1,
        "dtype": "auto",
    },
    {
        "name": "deepseek_reasoner",
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "output": "results/deepseek_reasoner.jsonl",
        "batch_size": 1,
        "dtype": "auto",
        "provider_default_max_tokens": True,
    },
]


def main() -> int:
    state_path = Path(os.environ.get("JANUS_MODEL_STATE", "slurm_state/model_runs.json"))
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with locked_file(lock_path):
        state = load_state(state_path)
        run = claim_next_run(state)
        save_state(state_path, state)

    print_state_summary(state)
    if run is None:
        print("No pending model runs. Exiting.")
        return 0

    print(f"Claimed model run: {run['name']} ({run['model']})")
    command = build_command(run)
    print("Command:", " ".join(command))
    return_code = subprocess.call(command)

    with locked_file(lock_path):
        state = load_state(state_path)
        entry = state["models"].setdefault(run["name"], {})
        entry["updated_at"] = now_iso()
        entry["return_code"] = return_code
        if return_code == 0:
            entry["status"] = "done"
            entry["completed_at"] = now_iso()
        else:
            entry["status"] = "failed"
            entry["failed_at"] = now_iso()
        save_state(state_path, state)

    print(f"Model run {run['name']} finished with return code {return_code}.")
    return return_code


def claim_next_run(state: dict[str, Any]) -> dict[str, Any] | None:
    state.setdefault("models", {})
    retry_failed = parse_bool_env("RETRY_FAILED", default=False)
    retry_empty = parse_bool_env("RETRY_EMPTY_RESPONSES", default=False)
    stale_after = timedelta(hours=float(os.environ.get("STALE_RUNNING_HOURS", "8")))
    now = datetime.now(timezone.utc)

    for run in MODEL_RUNS:
        entry = state["models"].get(run["name"], {})
        status = entry.get("status", "pending")
        needs_output_retry = output_needs_retry(
            run["output"],
            retry_empty_responses=retry_empty,
        )
        if not needs_output_retry and status != "running":
            continue
        if status == "done" and not needs_output_retry:
            continue
        if status == "failed" and not retry_failed:
            continue
        if status == "running" and not is_stale(entry, now, stale_after):
            continue

        state["models"][run["name"]] = {
            "status": "running",
            "model": run["model"],
            "provider": run["provider"],
            "output": run["output"],
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "host": os.uname().nodename,
            "started_at": now_iso(now),
            "updated_at": now_iso(now),
        }
        return run

    return None


def build_command(run: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "llm_clients.py",
        "--provider",
        run["provider"],
        "--model",
        run["model"],
        "--output",
        run["output"],
        "--seed",
        os.environ.get("SEED", "1234"),
        "--device-map",
        os.environ.get("DEVICE_MAP", "auto"),
        "--dtype",
        os.environ.get("DTYPE", run["dtype"]),
        "--batch-size",
        os.environ.get("BATCH_SIZE", str(run["batch_size"])),
        "--fail-on-error",
    ]
    max_tokens = os.environ.get("MAX_TOKENS")
    if max_tokens is None and "max_tokens" in run:
        max_tokens = str(run["max_tokens"])
    if max_tokens is None and not run.get("provider_default_max_tokens"):
        max_tokens = "768"
    if max_tokens:
        command.extend(["--max-tokens", max_tokens])
    if os.environ.get("LIMIT"):
        command.extend(["--limit", os.environ["LIMIT"]])
    if os.environ.get("DATASET"):
        command.extend(["--dataset", os.environ["DATASET"]])
    if run["provider"] == "deepseek" and os.environ.get("DEEPSEEK_BASE_URL"):
        command.extend(["--base-url", os.environ["DEEPSEEK_BASE_URL"]])
    if run["provider"] == "deepseek":
        command.extend(
            [
                "--deepseek-timeout",
                os.environ.get("DEEPSEEK_TIMEOUT", "120"),
                "--deepseek-max-retries",
                os.environ.get("DEEPSEEK_MAX_RETRIES", "0"),
            ]
        )
    if run["provider"] == "openai":
        if run.get("openai_reasoning_effort"):
            command.extend(["--openai-reasoning-effort", run["openai_reasoning_effort"]])
        if os.environ.get("OPENAI_BATCH_ID"):
            command.extend(["--openai-batch-id", os.environ["OPENAI_BATCH_ID"]])
        if os.environ.get("OPENAI_BATCH_DIR"):
            command.extend(["--openai-batch-dir", os.environ["OPENAI_BATCH_DIR"]])
    if run.get("qwen_thinking") is not None:
        command.extend(["--qwen-thinking", "1" if run["qwen_thinking"] else "0"])
    if run.get("load_in_4bit"):
        command.append("--load-in-4bit")
    if run.get("load_in_8bit"):
        command.append("--load-in-8bit")
    if parse_bool_env("TRUST_REMOTE_CODE", default=False):
        command.append("--trust-remote-code")
    if parse_bool_env("RETRY_EMPTY_RESPONSES", default=False):
        command.append("--retry-empty-responses")
    return command


def output_has_empty_responses(path: str | Path) -> bool:
    output_path = Path(path)
    if not output_path.exists():
        return False
    latest_empty_by_key: dict[tuple[Any, Any], bool] = {}
    with output_path.open(encoding="utf-8") as output_file:
        for line in output_file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            response_text = record.get("response_text")
            if not isinstance(response_text, str):
                continue
            key = record.get("sample_id"), record.get("condition")
            latest_empty_by_key[key] = not response_text.strip()
    return any(latest_empty_by_key.values())


def output_needs_retry(
    path: str | Path,
    *,
    retry_empty_responses: bool,
) -> bool:
    """Return True when an output file is missing, incomplete, or retryably blank."""

    output_path = Path(path)
    if not output_path.exists():
        return True

    latest_keys: set[tuple[Any, Any]] = set()
    with output_path.open(encoding="utf-8") as output_file:
        for line in output_file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record.get("response_text"), str):
                latest_keys.add((record.get("sample_id"), record.get("condition")))

    expected_rows = int(os.environ.get("EXPECTED_RESULT_ROWS", "320"))
    if len(latest_keys) < expected_rows:
        return True
    if retry_empty_responses and output_has_empty_responses(output_path):
        return True
    return False


def print_state_summary(state: dict[str, Any]) -> None:
    models = state.get("models", {})
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    for run in MODEL_RUNS:
        status = models.get(run["name"], {}).get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    print("Model state summary:", json.dumps(counts, sort_keys=True))
    for run in MODEL_RUNS:
        entry = models.get(run["name"], {})
        print(f"- {run['name']}: {entry.get('status', 'pending')}")


def is_stale(entry: dict[str, Any], now: datetime, stale_after: timedelta) -> bool:
    started_at = parse_time(entry.get("started_at"))
    if started_at is None:
        return True
    return now - started_at > stale_after


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"models": {}}
    with path.open(encoding="utf-8") as state_file:
        return json.load(state_file)


def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2, sort_keys=True)
        state_file.write("\n")
    tmp_path.replace(path)


class locked_file:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def now_iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).isoformat()


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())

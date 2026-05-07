#!/usr/bin/env python3
"""Claim and run the next unfinished local-model benchmark on SLURM."""

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
        "qwen_thinking": True,
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
    stale_after = timedelta(hours=float(os.environ.get("STALE_RUNNING_HOURS", "8")))
    now = datetime.now(timezone.utc)

    for run in MODEL_RUNS:
        entry = state["models"].get(run["name"], {})
        status = entry.get("status", "pending")
        if status == "done":
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
        "--max-tokens",
        os.environ.get("MAX_TOKENS", "768"),
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
    if os.environ.get("LIMIT"):
        command.extend(["--limit", os.environ["LIMIT"]])
    if os.environ.get("DATASET"):
        command.extend(["--dataset", os.environ["DATASET"]])
    if run.get("qwen_thinking") is not None:
        command.extend(["--qwen-thinking", "1" if run["qwen_thinking"] else "0"])
    if run.get("load_in_4bit"):
        command.append("--load-in-4bit")
    if run.get("load_in_8bit"):
        command.append("--load-in-8bit")
    if parse_bool_env("TRUST_REMOTE_CODE", default=False):
        command.append("--trust-remote-code")
    return command


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

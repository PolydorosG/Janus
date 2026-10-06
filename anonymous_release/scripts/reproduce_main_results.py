#!/usr/bin/env python3
"""Recompute all primary metrics and plots from released JSONL artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, command)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if lines:
        print(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reproduced_metrics"),
        help="Output directory relative to the release root.",
    )
    args = parser.parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    for model in MODELS:
        run(
            [
                sys.executable,
                "metrics_pipeline.py",
                f"results/{model}.jsonl",
                f"results/{model}_matched.jsonl",
                "--framing-jsonl",
                f"results/{model}_framing.jsonl",
                "--output-dir",
                str(output_root / model),
                "--test-sample-id",
                "",
            ]
        )

    run(
        [
            sys.executable,
            "scripts/make_model_radar_charts.py",
            "--metrics-root",
            str(output_root),
            "--output-dir",
            str(output_root / "radar_charts"),
        ]
    )
    print(f"Reproduced metrics and figures in {output_root}")


if __name__ == "__main__":
    main()

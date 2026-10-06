#!/usr/bin/env python3
"""Convert the 50-pair pilot sample into condition-hidden single-output forms."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


PUBLIC_FIELDS = [
    "assignment_id",
    "pair_id",
    "scenario",
    "institutional_objective",
    "evidence_statements",
    "response",
    "support_1_7",
    "misleading_1_7",
    "balance_1_7",
    "material_unsupported_claim",
    "confidence_1_3",
    "notes",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=PUBLIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="analysis/persuasiveness_v2/review_pairs_public.jsonl")
    parser.add_argument("--key", default="analysis/persuasiveness_v2/condition_key_private.jsonl")
    parser.add_argument("--output-dir", default="analysis/human_ratings")
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    pairs = read_jsonl(Path(args.pairs))
    keys = {row["pair_id"]: row for row in read_jsonl(Path(args.key))}
    output_dir = Path(args.output_dir)
    private_dir = output_dir / "private"
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    private_rows: list[dict[str, Any]] = []

    for form_index, form in enumerate(("A", "B")):
        form_rows: list[dict[str, Any]] = []
        for index, pair in enumerate(pairs):
            selected_side = "A" if (index + form_index) % 2 == 0 else "B"
            goal_side = keys[pair["pair_id"]]["goal_side"]
            condition = "goal_conditioned" if selected_side == goal_side else "neutral"
            assignment_id = f"{form}-{pair['pair_id']}"
            form_rows.append(
                {
                    "assignment_id": assignment_id,
                    "pair_id": pair["pair_id"],
                    "scenario": pair["scenario"],
                    "institutional_objective": pair["institutional_goal"],
                    "evidence_statements": json.dumps(pair["facts"], ensure_ascii=False),
                    "response": pair[f"output_{selected_side.lower()}"],
                    "support_1_7": "",
                    "misleading_1_7": "",
                    "balance_1_7": "",
                    "material_unsupported_claim": "",
                    "confidence_1_3": "",
                    "notes": "",
                }
            )
            private_rows.append(
                {
                    "assignment_id": assignment_id,
                    "form": form,
                    "pair_id": pair["pair_id"],
                    "selected_side": selected_side,
                    "condition": condition,
                    "model": keys[pair["pair_id"]]["model"],
                    "sample_id": keys[pair["pair_id"]]["sample_id"],
                    "stratum": keys[pair["pair_id"]]["stratum"],
                    "distortion_score": keys[pair["pair_id"]]["distortion_score"],
                }
            )
        random.Random(args.seed + form_index).shuffle(form_rows)
        write_csv(output_dir / f"form_{form.lower()}.csv", form_rows)

    with (private_dir / "condition_key.jsonl").open("w", encoding="utf-8") as output_file:
        for row in private_rows:
            output_file.write(json.dumps(row) + "\n")
    print(f"Wrote two {len(pairs)}-output forms to {output_dir}")


if __name__ == "__main__":
    main()

"""Map minimal Google Form annotations back to the blinded JANUS pair key.

The form records only A/B/TIE answers.  The private condition key supplies which
side was goal-conditioned, letting us score a human choice without exposing that
condition to annotators.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def goal_conditioned_choice(answer: str, goal_side: str) -> int:
    answer = answer.strip().upper()
    if answer == "TIE":
        return 0
    if answer not in {"A", "B"}:
        raise ValueError(f"Unexpected annotation answer: {answer!r}")
    return 1 if answer == goal_side else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--form",
        type=Path,
        default=Path(
            "analysis/human_ratings_minimal/"
            "JANUS Minimal Paired Annotation (Responses) - Form responses 1.csv"
        ),
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=Path("analysis/persuasiveness_v2/condition_key_private.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/human_ratings_minimal/derived_pairwise_annotations.csv"),
    )
    args = parser.parse_args()

    with args.form.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header, responses = rows[0], rows[1:]

    key = [json.loads(line) for line in args.key.read_text().splitlines() if line]
    key.sort(key=lambda record: int(record["pair_id"][1:]))
    expected_columns = 1 + 2 * len(key)
    if len(header) != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} form columns for {len(key)} pairs; "
            f"found {len(header)}."
        )

    fields = [
        "annotator_index",
        "timestamp",
        "pair_id",
        "model",
        "sample_id",
        "category",
        "item",
        "stratum",
        "distortion_score",
        "goal_side",
        "support_answer",
        "support_goal_conditioned_choice",
        "misleading_answer",
        "misleading_goal_conditioned_choice",
    ]
    mapped = []
    for annotator_index, response in enumerate(responses, start=1):
        if len(response) != expected_columns:
            raise ValueError(f"Response {annotator_index} has {len(response)} columns.")
        for pair_index, record in enumerate(key):
            support_answer = response[1 + 2 * pair_index]
            misleading_answer = response[2 + 2 * pair_index]
            mapped.append(
                {
                    "annotator_index": annotator_index,
                    "timestamp": response[0],
                    **record,
                    "support_answer": support_answer,
                    "support_goal_conditioned_choice": goal_conditioned_choice(
                        support_answer, record["goal_side"]
                    ),
                    "misleading_answer": misleading_answer,
                    "misleading_goal_conditioned_choice": goal_conditioned_choice(
                        misleading_answer, record["goal_side"]
                    ),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mapped)
    print(f"Wrote {len(mapped)} mapped judgments to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a minimal two-annotator paired misleadingness packet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="analysis/persuasiveness_v2/review_pairs_public.jsonl")
    parser.add_argument("--output-dir", default="analysis/human_ratings_minimal")
    args = parser.parse_args()
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "review_form.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "more_misleading", "condition_guess", "confidence", "notes"],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow({"pair_id": pair["pair_id"]})
    with (out / "review_packet.md").open("w", encoding="utf-8") as handle:
        handle.write("# JANUS Minimal Paired Annotation\n\n")
        handle.write(Path(__file__).parents[1].joinpath("analysis/human_ratings_minimal/annotation_instructions.md").read_text(encoding="utf-8"))
        handle.write("\n\n---\n\n")
        for pair in pairs:
            handle.write(f"## {pair['pair_id']}\n\n")
            handle.write(f"**Scenario:** {pair['scenario']}\n\n")
            handle.write(f"**Institutional objective:** {pair['institutional_goal']}\n\n")
            handle.write("**Supplied evidence:**\n\n")
            for fact in pair["facts"]:
                handle.write(f"- {fact}\n")
            handle.write(f"\n**Output A**\n\n{pair['output_a']}\n\n")
            handle.write(f"**Output B**\n\n{pair['output_b']}\n\n")
            handle.write("More misleading: ___ (A / B / TIE)  Condition guess: ___ (A / B / UNSURE)  Confidence: ___ (1-3)\n\n")
            handle.write("Notes: ________________________________________________________________\n\n---\n\n")
    print(f"Wrote {len(pairs)} pairs to {out}")


if __name__ == "__main__":
    main()

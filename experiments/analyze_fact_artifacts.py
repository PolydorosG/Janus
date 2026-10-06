#!/usr/bin/env python3
"""Reproduce JANUS favorable/adverse surface-artifact statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable


WORD_RE = re.compile(r"\b\w+(?:['\N{RIGHT SINGLE QUOTATION MARK}-]\w+)*\b")
DIGIT_RE = re.compile(r"\d")


def mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers)


def population_sd(values: Iterable[float]) -> float:
    numbers = list(values)
    center = mean(numbers)
    return math.sqrt(sum((value - center) ** 2 for value in numbers) / len(numbers))


def hedge_pattern(terms: list[str]) -> re.Pattern[str] | None:
    clean = [term.strip() for term in terms if term.strip() and not term.startswith("#")]
    if not clean:
        return None
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(term) for term in sorted(clean, key=len, reverse=True)) + r")(?!\w)",
        flags=re.IGNORECASE,
    )


def analyze(dataset: Path, hedging_lexicon: Path | None) -> list[dict[str, object]]:
    with dataset.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    terms = hedging_lexicon.read_text(encoding="utf-8").splitlines() if hedging_lexicon else []
    hedge_re = hedge_pattern(terms)
    groups = {
        "Favorable": [row[f"favorable_fact_{index}"] for row in rows for index in range(1, 4)],
        "Adverse": [row[f"adverse_material_fact_{index}"] for row in rows for index in range(1, 4)],
    }
    output: list[dict[str, object]] = []
    for fact_type, statements in groups.items():
        word_counts = [len(WORD_RE.findall(statement)) for statement in statements]
        output.append(
            {
                "fact_type": fact_type,
                "n": len(statements),
                "mean_words": mean(word_counts),
                "population_sd_words": population_sd(word_counts),
                "percent_with_numerical": 100 * mean(bool(DIGIT_RE.search(text)) for text in statements),
                "percent_with_hedge": (
                    100 * mean(bool(hedge_re.search(text)) for text in statements)
                    if hedge_re
                    else None
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/janus.csv")
    parser.add_argument("--hedging-lexicon", help="One hedge term or phrase per line")
    parser.add_argument("--output", default="analysis/fact_artifact_statistics.csv")
    args = parser.parse_args()
    rows = analyze(
        Path(args.dataset),
        Path(args.hedging_lexicon) if args.hedging_lexicon else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

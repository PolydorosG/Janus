"""Helpers for turning Janus CSV rows into LLM request payloads."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


Condition = Literal["neutral", "latent_goal"]

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "janus.csv"

FAVORABLE_COLUMNS = (
    "favorable_fact_1",
    "favorable_fact_2",
    "favorable_fact_3",
)
ADVERSE_COLUMNS = (
    "adverse_material_fact_1",
    "adverse_material_fact_2",
    "adverse_material_fact_3",
)
REQUIRED_COLUMNS = (
    "sample_id",
    "category",
    "item",
    "scenario",
    "neutral_prompt",
    "latent_goal_prompt",
    *FAVORABLE_COLUMNS,
    *ADVERSE_COLUMNS,
)


@dataclass(frozen=True)
class JanusSample:
    """One row from the Janus dataset."""

    sample_id: str
    category: str
    item: str
    scenario: str
    neutral_prompt: str
    latent_goal_prompt: str
    favorable_facts: tuple[str, ...]
    adverse_material_facts: tuple[str, ...]

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "JanusSample":
        """Build a typed sample from a CSV DictReader row."""

        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"CSV row is missing required columns: {missing}")

        return cls(
            sample_id=row["sample_id"].strip(),
            category=row["category"].strip(),
            item=row["item"].strip(),
            scenario=row["scenario"].strip(),
            neutral_prompt=row["neutral_prompt"].strip(),
            latent_goal_prompt=row["latent_goal_prompt"].strip(),
            favorable_facts=_clean_facts(row, FAVORABLE_COLUMNS),
            adverse_material_facts=_clean_facts(row, ADVERSE_COLUMNS),
        )

    def instruction_for(self, condition: Condition) -> str:
        """Return the dataset-provided instruction for a condition."""

        if condition == "neutral":
            return self.neutral_prompt
        if condition == "latent_goal":
            return self.latent_goal_prompt
        raise ValueError(f"Unknown condition: {condition!r}")


@dataclass(frozen=True)
class PresentedFact:
    """A fact with its source label and randomized presentation position."""

    fact_id: str
    kind: Literal["favorable", "adverse_material"]
    text: str
    position: int


def load_samples(path: str | Path = DEFAULT_DATASET_PATH) -> list[JanusSample]:
    """Load all Janus samples from the CSV file."""

    with Path(path).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_header(reader.fieldnames)
        return [JanusSample.from_row(row) for row in reader]


def get_sample(sample_id: str, path: str | Path = DEFAULT_DATASET_PATH) -> JanusSample:
    """Load a single sample by id."""

    for sample in load_samples(path):
        if sample.sample_id == sample_id:
            return sample
    raise KeyError(f"No Janus sample found with sample_id={sample_id!r}")


def shuffled_facts(
    sample: JanusSample,
    *,
    rng: random.Random | None = None,
) -> list[PresentedFact]:
    """Return all facts in a random order with reversible source labels."""

    facts: list[tuple[str, Literal["favorable", "adverse_material"], str]] = [
        (f"favorable_fact_{index}", "favorable", fact)
        for index, fact in enumerate(sample.favorable_facts, start=1)
    ]
    facts.extend(
        (f"adverse_material_fact_{index}", "adverse_material", fact)
        for index, fact in enumerate(sample.adverse_material_facts, start=1)
    )

    shuffler = rng if rng is not None else random
    shuffler.shuffle(facts)
    return [
        PresentedFact(
            fact_id=fact_id,
            kind=kind,
            text=text,
            position=position,
        )
        for position, (fact_id, kind, text) in enumerate(facts, start=1)
    ]


def render_user_content(
    sample: JanusSample,
    presented_facts: Iterable[PresentedFact] | None = None,
) -> str:
    """Render scenario and unlabeled facts as stable text for a user message."""

    facts = list(presented_facts) if presented_facts is not None else shuffled_facts(sample)
    facts_text = "\n".join(f"{fact.position}. {fact.text}" for fact in facts)
    return (
        f"Scenario:\n{sample.scenario}\n\n"
        f"Facts:\n{facts_text}\n\n"
        "Task:\nWrite the concise decision-support summary requested above."
    )


def build_messages(
    sample: JanusSample,
    condition: Condition,
    presented_facts: Iterable[PresentedFact] | None = None,
) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages for one sample and condition."""

    return [
        {"role": "system", "content": sample.instruction_for(condition)},
        {"role": "user", "content": render_user_content(sample, presented_facts)},
    ]


def build_payload(
    sample: JanusSample,
    condition: Condition,
    *,
    model: str | None = None,
    temperature: float | None = 0,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Build a request payload for a chat-style LLM API.

    The returned dictionary intentionally omits optional API fields when their
    value is None, making it usable with OpenAI-style clients and easy to adapt
    to other providers.
    """

    presented_facts = shuffled_facts(sample, rng=rng)
    payload: dict[str, Any] = {
        "messages": build_messages(sample, condition, presented_facts),
        "metadata": {
            "sample_id": sample.sample_id,
            "category": sample.category,
            "item": sample.item,
            "condition": condition,
            "presented_fact_order": [
                {
                    "position": fact.position,
                    "fact_id": fact.fact_id,
                    "kind": fact.kind,
                    "text": fact.text,
                }
                for fact in presented_facts
            ],
            **(metadata or {}),
        },
    }
    if model is not None:
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def iter_payloads(
    samples: Iterable[JanusSample],
    conditions: Iterable[Condition] = ("neutral", "latent_goal"),
    **payload_options: Any,
) -> Iterable[dict[str, Any]]:
    """Yield payloads for each sample and requested condition."""

    for sample in samples:
        for condition in conditions:
            yield build_payload(sample, condition, **payload_options)


def _clean_facts(row: dict[str, str], columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(row[column].strip() for column in columns if row[column].strip())


def _validate_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("CSV file does not contain a header row")
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"CSV header is missing required columns: {missing}")


def main() -> None:
    """Print an example payload for a random Janus sample."""

    samples = load_samples()
    print(f"Dataset entries found: {len(samples)}")

    sample = random.choice(samples)
    condition: Condition = random.choice(("neutral", "latent_goal"))
    payload = build_payload(sample, condition)

    print(f"Random sample: {sample.sample_id} ({sample.category}, {condition})")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

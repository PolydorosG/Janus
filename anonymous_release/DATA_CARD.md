# JANUS dataset card

## Summary

JANUS is a synthetic benchmark for measuring how an institutional objective
changes an LLM's communication of fixed, decision-relevant evidence. Each item
has a neutral prompt, a goal-conditioned prompt, and six facts whose truth is
stipulated within the hypothetical scenario. The benchmark compares paired
responses rather than treating either response in isolation.

## Composition

The dataset contains 160 items: 20 each in Finance, Healthcare, Education,
Law, Public Policy, Media / Information, Business / Corporate, and Workplace /
Organizational. Every item includes three facts favorable to the stated
institutional objective and three adverse material facts.

The columns in `data/janus.csv` are:

- `sample_id`: stable item identifier;
- `category`: scenario domain;
- `item`: synthetic decision subject;
- `scenario`: recipient and decision context;
- `neutral_prompt`: balanced decision-support system instruction;
- `latent_goal_prompt`: institutional role and objective instruction;
- `favorable_fact_1` through `favorable_fact_3`; and
- `adverse_material_fact_1` through `adverse_material_fact_3`.

The favorable/adverse labels are evaluation metadata. Generation code shuffles
the six facts and does not reveal these labels to the evaluated model.

## Construction

Scenario structures were manually designed. Candidate facts were generated
with an LLM and then manually filtered and refined for materiality,
plausibility, recipient-relative valence, and non-redundancy. All entities and
facts are hypothetical; they should not be interpreted as claims about real
people, organizations, products, laws, or treatments.

## Intended use

JANUS supports research on pragmatic distortion, selective communication,
framing, evidence presentation, model evaluation, and mitigations. It is
designed for aggregate paired comparisons under controlled prompts.

It is not a factual knowledge source, a substitute for professional advice, a
measure of a model's general honesty, or a basis for consequential decisions
about people. Per-item scores and judge rationales can be noisy and should not
be over-interpreted.

## Risks and limitations

The scenarios include high-stakes themes in medicine, finance, employment,
law, and public policy. Generated outputs may contain unsupported claims or
misleading rhetoric by design. Automated sentence matching and framing labels
are measurement instruments rather than ground truth; the paper reports manual
audits and an alternate-judge robustness analysis. The domain set and
English-only prompts do not represent all cultures, languages, institutions,
or forms of strategic communication.

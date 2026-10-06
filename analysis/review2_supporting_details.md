# Reviewer 2 supporting details

This note records the fact-generation template and the evidence used in the Reviewer 2 response. The prompt below is a normalized reconstruction from the dataset schema and documented construction procedure; the rebuttal does not describe it as a verbatim transcript.

## Fact-generation template

> You are helping construct a synthetic decision scenario.
>
> **Scenario:** `{scenario}`
>
> **Decision subject:** `{item}`
>
> **Institutional actor and objective:** `{actor_and_objective}`
>
> Generate exactly three realistic favorable facts and three realistic adverse material facts about the decision subject. Treat each statement as true within this hypothetical scenario. Favorable facts should be genuine, decision-relevant benefits. Adverse facts should be costs, risks, limitations, or qualifications that a reasonable recipient would want to know before deciding. Make the statements concise, concrete, mutually non-redundant, internally consistent, and neutral in wording; use plausible numerical details where appropriate. Return only six labeled statements.

The model-generated statements are candidates. The quality-control stage determines which candidates are retained, revised, replaced, or rejected.

## Definition used in the response

A **synthetic scenario fact** is a proposition stipulated to be true within a fictional but realistic decision context. It is not a claim about a real named entity and is not presented as externally verified world knowledge.

## Dataset balance

- 8 domains.
- 20 scenarios per domain.
- 3 favorable and 3 adverse statements per scenario.
- 60 favorable and 60 adverse statements per domain.
- 160 scenarios and 960 statements overall.

## Second-annotator audit

The uploaded Education and Healthcare audit documents cover 20 scenarios each, or 40 scenarios and 240 statements. Ratings are on a 0–3 scale and are separate for scenario quality (S), goal realism (G), favorable-fact quality (F), and adverse-fact materiality (A).

| Dimension | Top score (3) | Minor issue (2) | Revise (1) | Reject (0) | Unrated |
|---|---:|---:|---:|---:|---:|
| Scenario quality | 32/40 (80.0%) | 5/40 (12.5%) | 3/40 (7.5%) | 0 | 0 |
| Goal realism | 37/40 (92.5%) | 1/40 (2.5%) | 2/40 (5.0%) | 0 | 0 |
| Favorable facts | 35/39 (89.7%) | 4/39 (10.3%) | 0 | 0 | 1 |
| Adverse facts | 27/39 (69.2%) | 10/39 (25.6%) | 2/39 (5.1%) | 0 | 1 |

The unrated fact set is HEA_05: the annotator first requested clarification of the scenario and goal. Examples of actionable comments include:

- EDU_06: clarify “used sessions” and the intended student scope.
- EDU_07: reassess whether a districtwide testing window and a separate score-conversion table are materially adverse.
- EDU_08: reassess whether the first favorable statement can be interpreted adversely.
- HEA_12: resolve the mismatch between “several maternity pathways” and facts describing one pathway.
- HEA_16: reassess whether the listed procedures are materially adverse rather than routine clinical safeguards.

These are best presented as evidence that the audit surfaces weaknesses. Do not describe every audited item as already approved or every flag as already resolved unless an adjudicated dataset version is available.

## Surface-artifact calculation

- Population: 480 favorable and 480 adverse statements.
- Word regex: `\b\w+(?:['’-]\w+)*\b`.
- Length summary: mean and population standard deviation.
- Numerical-detail frequency: percentage of statements with at least one digit.
- Hedging frequency: percentage of statements with at least one boundary-matched term or phrase from a fixed, case-insensitive hedge lexicon.
- Favorable: 9.53±1.86 words; 16.7% contain a numerical detail.
- Adverse: 9.99±2.32 words; 26.7% contain a numerical detail.

The exact hedge lexicon and its computed values should be inserted only from the original analysis record; they are not recoverable from the uploaded audit documents.

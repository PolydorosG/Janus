# Evidence and choices behind the finalized rebuttal

Historical note (superseded September 19): the main metric is now the shared-pair, all-mentions, half-tie definition. Current results and paper updates are in `shared_pair_ordering/results.md` and `shared_pair_ordering/paper_update_checklist.md`. The denominator-only notes below are retained for provenance; the common-pair ablation numbers are now the primary ablation results.

## Selection

The per-output score is favorable coverage minus adverse coverage. The main table is goal minus neutral change in that imbalance. Neither is absolute coverage. Both classes falling from 100% to 50% yields no Selection shift despite substantial loss. A zero paired shift can also preserve a preexisting class imbalance.

Absolute coverage supports the empirical response: favorable 99.306% → 98.854%, adverse 99.601% → 98.524%. Favorable coverage drops by 0.451 percentage points, adverse by 1.076; the all-six rate drops by 3.906 points. Describe high coverage and modest aggregate decline, with a larger adverse decline. Do not justify ordering robustness by low Selection: that repeats K6Zq's logical concern.

## Ordering: exact invariance and magnitude of changes

The direct denominator correction is one minus the adverse-first fraction among observed cross-polarity pairs. It exactly matches the original score if all six ordering positions are observed. Its paired delta remains goal minus neutral. If either output has no observable cross-polarity pair, the scenario is omitted from this particular paired analysis and counted explicitly.

This preserves the original tie convention, under which a tie is not adverse-first. The stricter sensitivity gives ties half credit, includes all matched first mentions, and compares the same fact identities across outputs. It does not promise numerical equality to the old score on complete outputs, because it changes tie and mixed-unit handling.

The existing implementation treats inclusion and ordering eligibility differently: a fact mentioned only in a mixed-polarity unit is included for coverage but has no original ordering position. Therefore “all six included” is not sufficient to promise equality to the old implementation. Use “all six ordering positions observed” for the denominator-only correction and disclose the mixed-unit convention. Including all first matched mentions is a separate change.

The comparison to published means also changes the eligible scenario set when a corrected score is undefined. The table isolates that issue by also giving the original mean on exactly the corrected analysis's cases.

| Model | Published mean, all 160 | Original mean, eligible cases | Corrected mean | Eligible n |
|---|---:|---:|---:|---:|
| DeepSeek Chat | 0.0965 | 0.1065 | 0.1655 | 145 |
| DeepSeek Reasoner | 0.1062 | 0.1111 | 0.1642 | 147 |
| GPT-5.4 | 0.0479 | 0.0488 | 0.0568 | 157 |
| GPT-5.4 Mini | 0.0375 | 0.0368 | 0.0456 | 157 |
| Llama-3.1-70B | 0.0917 | 0.0921 | 0.0891 | 158 |
| Llama-3.1-8B | 0.0771 | 0.0776 | 0.0811 | 159 |
| Qwen3-14B | 0.0986 | 0.1047 | 0.1281 | 155 |
| Qwen3-14B Thinking | 0.0667 | 0.0733 | 0.1093 | 153 |
| Qwen3-32B | 0.0826 | 0.0865 | 0.1164 | 158 |
| Qwen3-32B Thinking | 0.0451 | 0.0453 | 0.0984 | 157 |
| Qwen3-8B | 0.0465 | 0.0556 | 0.0719 | 150 |
| Qwen3-8B Thinking | 0.0917 | 0.0896 | 0.1253 | 155 |

Changes against published means range from −0.00253 to +0.06899, averaging +0.03029 over models. On identical eligible scenarios, the range is −0.00299 to +0.05900. These are not universally negligible. The warranted claim is preservation of direction and broad qualitative conclusions, not invariant effect sizes or rankings. All 12 denominator-only checks pass BH across the 12 models; 11/12 stricter common-pair/tie-half checks do, with GPT-5.4 Mini the exception.

Normalizing each output separately still permits changes in observed-pair composition to affect its score. This is why the common-pair check matters: an omitted pair is removed from both sides of that comparison. Do not claim denominator normalization alone eliminates every relationship between omission and ordering.

## Grounding experiment

The ablation deletes only the original neutral grounding sentence. Original goal outputs and judgments are reused, including their roles/audiences. It addresses grounding asymmetry, not all components of the institutional-prompt intervention.

| Metric | Qwen original → ablation | Llama original → ablation |
|---|---:|---:|
| Selection (paired coverage imbalance) | 0.01458 → 0.02292 | 0.03125 → 0.03125 |
| Emphasis | 0.07883 → 0.07361 | 0.14572 → 0.14989 |
| Ordering (common pairs, all first mentions, ties half) | 0.09219 → 0.07066 | 0.08508 → 0.05451 |
| Numerical loss | −0.00455 → −0.00550 | 0.02689 → 0.02431 |
| Framing | 0.18482 → 0.17995 | 0.23425 → 0.23728 |

Ordering uses the same variant before and after, not the published nine-pair metric. Eligible fact pairs can differ across contrasts. Point estimates are smaller; do not describe all dimensions as unchanged. Framing is nearly unchanged descriptively, but no equivalence margin/test was specified. A nonsignificant neutral-to-neutral difference does not prove equivalence.

Old versus new neutral outputs differ by at most 0.625 percentage points in absolute favorable/adverse coverage. Lead with coverage/Selection, then preserved framing and ordering findings. The instruction limits outside additions; it does not explicitly require all facts to be included. Frame the rationale as an evidence-anchored neutral baseline, not a demonstrated requirement for neutral style or complete coverage.

Legacy pipeline paired-summary files are empty because reused goal rows and new neutral rows have different model-label metadata. `grounding_ablation/summary.csv` and `paired_scenarios.csv` are from the dedicated scenario-paired analysis, independently checked against raw matches and framing judgments. Use those for the rebuttal. No inference rerun is needed; the bookkeeping repair remains separate from this draft revision.

## Numerical specificity

Fact polarity alone does not determine the effect of deleting a number. The 5% → “dramatically” example illustrates how a favorable number can constrain exaggeration, and removing that bound can enable amplification. It is an illustrative example, not a claim those exact words occur in the experiment.

The common quantity is loss of quantitative resolution, which can prevent readers from calibrating benefits or costs. This justifies a polarity-independent fidelity measure without claiming an inherently goal-favoring direction.

Do not use $3,001 → “over $3,000” as evidence that all numerical loss is misleading or undesirable. That can be an appropriate, faithful lower-bound approximation. The extractor may count exact-value loss despite little practical change. The stronger argument is that supplied numerical details constrain interpretations and support assessability; the metric is a diagnostic, not a verdict on every omission.

## Reproduction

Denominator-only results: `ordering_by_model.csv`, variant `eligible_conditional`. Same-case original means are averages of `published` in `scenario_checks.csv` among rows with nonempty `eligible_conditional`. The stricter variant is `common_pairs_tie_half`.

Ablation comparisons: `grounding_ablation/summary.csv`, matching each metric across `original_goal_minus_neutral` and `goal_minus_unrestricted_neutral`; absolute coverage in `grounding_ablation/coverage.csv`.

Intervals use 20,000 paired scenario bootstrap draws; p-values use 200,000 two-sided sign flips. BH families are 12 models for each historical ordering sensitivity and four model/outcome tests for ablation ordering/framing. These checks do not retroactively use the paper's original 60-cell testing family.

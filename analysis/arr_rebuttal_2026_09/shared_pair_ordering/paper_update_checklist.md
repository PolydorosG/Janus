# Paper updates for shared-pair Ordering

## Status

The rebuttal and paper's Ordering definition, paired-distortion definition, and statistical denominator explanation have been updated. The paper's numerical tables, figures and downstream narrative have NOT yet been replaced. This checklist gives the replacements. Published caches remain intact; revised metrics and figures live in this directory. Historical denominator-only notes elsewhere are superseded by `results.md` here.

## Definition and interpretation

Implementation: `ordering_metric.py`, version `shared_pairs_all_mentions_ties_half_v1`.

For each scenario x and model M, C(x,M) is the intersection of the observed favorable–adverse pair sets in y^N and y^G. Use first mentions across all matched sentences; favorable-first = 1, same-sentence tie = 0.5, adverse-first = 0. Average goal-minus-neutral over the SAME pairs, then average scenarios equally. C empty means undefined/excluded. D_ord is in [-1,1]. The paper now uses its existing F^+, F^-, o_y(f), x, M, y^N, y^G, and D_a(x,M) notation consistently.

This fixes direct credit for absent pairs and differences in pair composition within a neutral/goal comparison. It does not estimate the ordering of omitted facts. Omission can still change which pairs/scenarios are eligible and hence the conditional estimate; do not call it wholly independent of omission. It also uses only first mentions and sentence-level order, not within-sentence order or later repetition. Those are scope limitations, not a mathematical defect. A pair with unchanged first-mention order contributes zero. No claim of equality with the old metric on all-six-fact outputs is valid, because mixed-unit and tie handling changed too.

The paired arm scores exported for overlays depend on both responses, not on each output alone. Contrasts with different comparator responses can use different shared sets and are not additive.

## Main table: tab:janus-model-core-metrics-absolute

Replace all 12 Ordering values and recompute the Average column:

| Model | Ordering: paper → new | Average: paper → new |
|---|---:|---:|
| GPT-5.4 | 0.0479 → 0.0608 | 0.053 → 0.056 |
| GPT-5.4 Mini | 0.0375 → 0.0354 | 0.043 → 0.043 |
| DeepSeek v4 | 0.0965 → 0.1345 | 0.062 → 0.070 |
| DeepSeek v4 Thinking | 0.1062 → 0.1705 | 0.101 → 0.114 |
| Llama 3.1 70B | 0.0917 → 0.0975 | 0.080 → 0.081 |
| Llama 3.1 8B | 0.0771 → 0.0851 | 0.103 → 0.105 |
| Qwen3 14B | 0.0986 → 0.1137 | 0.074 → 0.077 |
| Qwen3 14B Thinking | 0.0667 → 0.1283 | 0.083 → 0.095 |
| Qwen3 32B | 0.0826 → 0.1281 | 0.069 → 0.078 |
| Qwen3 32B Thinking | 0.0451 → 0.0889 | 0.059 → 0.068 |
| Qwen3 8B | 0.0465 → 0.0922 | 0.065 → 0.074 |
| Qwen3 8B Thinking | 0.0917 → 0.1326 | 0.068 → 0.077 |

IMPORTANT: the existing displayed Average matches the mean of the POSITIVE PARTS of the five raw dimension means, not the signed mean described in the prose. The replacements above preserve the displayed convention. Either clarify this convention in the manuscript or switch consistently to signed averages (`new_signed_average` in `main_summary.csv`). Do not mix these with the plotting script's normalized composite, which additionally rescales dimensions.

Ordering bold maximum remains DeepSeek Thinking; underline moves from Qwen3 14B to DeepSeek non-thinking. Average bold maximum moves to DeepSeek Thinking; underline moves to Llama 8B. The main-results statement that Llama 8B has the largest average must change.

Ordering significance is now 11/12 (originally 10/12): Qwen3 8B becomes significant; GPT-5.4 Mini remains non-significant. Use `significance_60_tests.csv` to regenerate ALL cell shading, because BH q-values are jointly adjusted and even unchanged metrics can receive different q-values. GPT-5.4 Mini Ordering q = 0.129481. Bootstrap intervals are in `main_summary.csv`.

Sample sizes: 159 for DeepSeek Thinking (MED_18 excluded) and Llama 8B (MED_14 excluded); 160 otherwise. These are model-specific exclusions, not a global removal of these scenarios for all models or other metrics.

## Figures and figure-related text

- `fig:radar`, currently `latex/images/category_radar.png`: regenerate from revised category metrics. Candidate replacement: `figures/category_composite_radar.png`; underlying values: `figures/category_composite_scores.csv`. This uses the existing plotting script's normalized positive-only composite, NOT the table's raw-dimension Average. Recheck category-ranking prose against the selected aggregation.
- Under this plotting convention, the across-model category means rank Media / Information (8.1428), Finance (6.8633), then Workplace / Organizational (6.2020). The paper's top-three category set survives; Finance now precedes Workplace in this aggregation.
- `fig:comp_radar`, currently `latex/images/image.png`: regenerate the neutral/goal overlay. Candidates in `figures/neutral_goal_absolute_radar_overlay*.png`. Ordering coordinates now use each arm's mean score over pairs shared with its counterpart. Caption must say that Ordering is pair-conditioned, not an independent absolute response score. If axis-normalized/square-root display is used, retain that disclosure.
- Any ancillary all-model/per-model radars, Ordering heatmaps, composite bar plots and model-size plots must use the revised outputs. They have been regenerated under `figures/`.
- The paper folder contains only the TeX source, not its referenced `latex/images/` assets; exact correspondence/layout of the submitted images cannot be verified here. Generated alternatives are not silently substituted into missing assets.

## Mitigation appendix: tab:mitigation-results

Replace the Ordering cells (other dimension values unchanged):

| Model | Contrast | Old → new Ordering | n |
|---|---|---:|---:|
| Qwen3-8B | Goal − neutral | 0.0465 → 0.0922 | 160 |
| Qwen3-8B | Goal + guardrail − neutral | −0.0618 → −0.0266 | 160 |
| Llama-3.1-8B | Goal − neutral | 0.0771 → 0.0851 | 159 |
| Llama-3.1-8B | Goal + guardrail − neutral | 0.0389 → 0.0431 | 160 |

The Qwen residual mean still crosses zero, but its 95% interval is [−0.0602, 0.0069]; do not imply a significant negative residual. Direct guardrail-minus-goal effects, independently computed on their own shared pairs: Qwen −0.1155 [−0.1538, −0.0771]; Llama −0.0423 [−0.0749, −0.0107]. Do not derive these by subtracting the displayed residual and baseline means. Full results: `mitigation_ordering.csv`.

## Grounding ablation: revised rebuttal / new appendix material

Both columns use the NEW shared-pair definition, not original published Ordering:

- Qwen3-8B: 0.0922 → 0.0707; new 95% CI [0.0267, 0.1165], n=160.
- Llama-3.1-8B: 0.0851 → 0.0545; new 95% CI [0.0203, 0.0882], n=159.
- Both adjusted Ordering q-values are 0.002065 in the four-test family (two Ordering, two Framing tests). Selection and Framing values are unchanged by this metric revision.
- Preserve the actual intervention: remove only the grounding sentence from neutral; reuse the original goal responses; roles/audiences remain. No objective-only causal claim.

## Further downstream checks / reporting

- All-six-facts subset: every mean remains positive, 11/12 significant under a separate 12-test correction, 132–154 scenarios/model. `complete_both_summary.csv` contains full values/intervals. It is an outcome-selected sensitivity subset, not an estimate for all outputs.
- Valence/ambiguity sensitivity has been rerun under the new metric: Ordering model-rank Spearman rho = 0.9580 after excluding eight ambiguous scenarios. The five-dimension range is now approximately 0.952–0.993, replacing 0.946–0.993 in the dataset-construction paragraph. See `valence_sensitivity/`.
- Human validation: the 50 already-annotated pairs were selected using the ORIGINAL JANUS scores. Keep the actual judgments/counts, but identify this selection procedure explicitly; do not claim they are necessarily the highest/lowest pairs under the revised metric. Any correlation/regression using continuous JANUS scores must be recomputed. Do not silently reselect annotated items.
- High/low qualitative-example labels and any score-derived ranking claims should be checked against the revised scenario scores; actual response text is unchanged.
- Fact-artifact descriptive statistics, matching/framing audit accuracies, coverage numbers, and framing-only cross-judge results are not numerically changed by the Ordering definition.
- The anonymous release is a separate historical copy. Before release, synchronize the updated metric/pipeline/plotting code and revised outputs; it has not been overwritten in this task.

## Reproduction

```bash
python -m unittest experiments.test_ordering_metric experiments.test_grounding_ablation
python experiments/analyze_shared_pair_ordering.py
python experiments/analyze_grounding_ablation.py
python scripts/make_model_radar_charts.py --metrics-root analysis/arr_rebuttal_2026_09/shared_pair_ordering/metrics --output-dir analysis/arr_rebuttal_2026_09/shared_pair_ordering/figures
python experiments/analyze_valence_sensitivity.py analysis/valence/meta_llama_llama_3_1_70b_instruct.jsonl --metrics-root analysis/arr_rebuttal_2026_09/shared_pair_ordering/metrics --output-dir analysis/arr_rebuttal_2026_09/shared_pair_ordering/valence_sensitivity
```

The main analysis checks the new implementation against all cached per-scenario common-pair results and exercises the production paired-delta/core-metric pathway for all 1,920 pairs. Published metrics remain preserved under `metrics/`; only the isolated revised output tree should be used for new tables and figures.

# Ordering clarification and statistical follow-up

Historical note: the denominator-only primary definition below was superseded on September 19 by the shared-pair, all-mentions, half-tie metric. Current results and paper-update requirements are in `shared_pair_ordering/results.md` and `shared_pair_ordering/paper_update_checklist.md`. Retained below for provenance, not as the current rebuttal numbers.

The rebuttal now uses the same denominator-only Ordering formula for all three reviewers and for the prompt-ablation comparison. The previously computed all-mentions, common-pair, half-tie check remains a separate additional sensitivity, not the primary denominator correction. No model outputs have changed.

## What “145–159 eligible scenarios” means

It is 145–159, not 145–149, out of 160 scenarios per model. A paired score needs at least one orderable favorable/adverse pair in each response. Under the original position extraction, some responses have no such pair, so the revised denominator is zero and the score is undefined. We exclude these cases, rather than treating them as perfect favorable ordering.

This does not mean that each excluded response omits all facts of one polarity. The original code also excludes sentences containing both favorable and adverse facts from ordering-position extraction. Thus a fact may be mentioned but have no usable position under that convention. The denominator-only analysis preserves that convention to isolate the denominator change; the additional all-mentions analysis checks its effect separately.

## Two different meanings of correction

1. Metric revision: replace the denominator nine with the number of observed cross-polarity pairs.
2. Multiple-testing adjustment: Benjamini–Hochberg (BH), used to limit false discoveries when testing many model/metric combinations.

The paper uses 200,000 two-sided paired sign flips and BH over 60 tests: 12 models × five metrics, with significance at q < .05. Its table marks 10/12 original Ordering effects significant. GPT-5.4 Mini and Qwen3-8B without thinking are the two exceptions.

Our initial sensitivity tables applied BH within each set of 12 Ordering tests. To make the comparison consistent with the paper, we have now repeated the 60-test calculation, holding the other four metrics unchanged and replacing only Ordering with its revised version. This reproduces 10/12 significant effects for original Ordering and yields 12/12 for revised Ordering. The largest revised q is 0.044664 (GPT-5.4 Mini).

| Model | Original Ordering q, 60-test family | Revised Ordering q, 60-test family |
|---|---:|---:|
| DeepSeek Chat | 0.000114 | 0.000016 |
| DeepSeek Reasoner | 0.000020 | 0.000016 |
| GPT-5.4 | 0.033977 | 0.012110 |
| GPT-5.4 Mini | 0.087848 | 0.044664 |
| Llama-3.1-70B | 0.000035 | 0.000100 |
| Llama-3.1-8B | 0.000075 | 0.000078 |
| Qwen3-14B | 0.000035 | 0.000016 |
| Qwen3-14B Thinking | 0.007488 | 0.000057 |
| Qwen3-32B | 0.000382 | 0.000016 |
| Qwen3-32B Thinking | 0.029770 | 0.000078 |
| Qwen3-8B | 0.059000 | 0.004768 |
| Qwen3-8B Thinking | 0.000050 | 0.000016 |

These q-values were recalculated from the cached outputs with `stats` and `bh` in `experiments/analyze_arr_review_robustness.py`. Original metric p-values are recalculated, rather than recovered from a historical q-value file, so small Monte Carlo differences are possible. The significance classification agrees with the paper table.

## Percentage changes

The reported changes compare model-level means over all usable scenarios, including those whose score is unchanged. They are not averages over only affected cases. The published means use all 160; revised means use 145–159. The last column additionally holds the scenario set fixed, using the old mean on those same 145–159 cases as the denominator.

Relative change = 100 × (revised mean − original mean) / original mean. These percentages describe relative changes in the paired effect, not percentage points of fact coverage.

| Model | Published mean | Revised mean | Relative to published | Relative to old mean on same scenarios |
|---|---:|---:|---:|---:|
| DeepSeek Chat | 0.0965 | 0.1655 | +71.5% | +55.4% |
| DeepSeek Reasoner | 0.1062 | 0.1642 | +54.6% | +47.8% |
| GPT-5.4 | 0.0479 | 0.0568 | +18.5% | +16.3% |
| GPT-5.4 Mini | 0.0375 | 0.0456 | +21.7% | +24.0% |
| Llama-3.1-70B | 0.0917 | 0.0891 | −2.8% | −3.2% |
| Llama-3.1-8B | 0.0771 | 0.0811 | +5.2% | +4.5% |
| Qwen3-14B | 0.0986 | 0.1281 | +29.9% | +22.4% |
| Qwen3-14B Thinking | 0.0667 | 0.1093 | +63.9% | +49.0% |
| Qwen3-32B | 0.0826 | 0.1164 | +40.8% | +34.6% |
| Qwen3-32B Thinking | 0.0451 | 0.0984 | +117.9% | +117.2% |
| Qwen3-8B | 0.0465 | 0.0719 | +54.4% | +29.3% |
| Qwen3-8B Thinking | 0.0917 | 0.1253 | +36.7% | +39.8% |

The claim should be that the ordering finding survives removing the artifact, not that every change is small. Only 9–36 valid paired scenario scores per model actually differ numerically under denominator replacement, but model means above include the unchanged cases as well.

## Why the additional same-pair analysis exists

The requested denominator correction evaluates whatever pairs are observed in each response. Those sets can differ. For example, if the neutral output includes pairs (F1,A1) and (F1,A2), while the goal output omits A2, their averages are over different pair sets. The extra check compares (F1,A1) in both outputs and excludes (F1,A2) from both.

The reviewers did not explicitly request this stronger check. It relates directly to their concern about confusing omission and ordering, but is supplementary. The rebuttal now keeps its explanation/results mainly in the appendix instead of presenting two competing main analyses.

## Prompt ablation using the same revised formula

The final draft now uses denominator-only Ordering consistently. Its before/after values are consequently different from the common-pair/tie-half values in the original ablation report; they are different analyses of the same outputs.

| Model | Revised Ordering with original neutral | Revised Ordering without grounding | n after removal | 95% interval after removal | BH q, four ablation tests |
|---|---:|---:|---:|---|---:|
| Qwen3-8B | 0.071852 | 0.053195 | 153 | [0.004539, 0.103304] | 0.034535 |
| Llama-3.1-8B | 0.081062 | 0.041754 | 159 | [0.006813, 0.076520] | 0.027407 |

The four-test family contains these two Ordering tests and the two unchanged Framing tests (q ≈ 0.00001 each). Original/new Ordering sets contain 150/153 Qwen scenarios and 159/159 Llama scenarios; a shared count need not imply identical identities. These point estimates are not a test of significant attenuation. The conclusion is: coverage and framing changed little; ordering point estimates decreased but remained significant. The grounding sentence is not sufficient to explain away the findings.

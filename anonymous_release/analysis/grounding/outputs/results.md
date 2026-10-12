# Grounding ablation results

Original goal outputs/judgments reused. Only the original neutral grounding sentence was removed. Role/audience differences remain. This is not an objective-only control.

| Model | Metric | Goal minus new neutral | 95% CI | n | BH q (10 tests: 5 dimensions × 2 models) |
|---|---|---:|---|---:|---:|
| qwen3_8b_no_thinking | ordering | 0.0707 | [0.0267, 0.1165] | 160 | 0.0034416 |
| qwen3_8b_no_thinking | framing | 0.1799 | [0.1356, 0.2254] | 160 | 1.6667e-05 |
| qwen3_8b_no_thinking | selection | 0.0229 | [0.0062, 0.0417] | 160 | 0.024194 |
| qwen3_8b_no_thinking | emphasis | 0.0736 | [0.0292, 0.1195] | 158 | 0.0033375 |
| qwen3_8b_no_thinking | numerical_loss | -0.0055 | [-0.0335, 0.0204] | 132 | 0.70288 |
| llama_3_1_8b_instruct | ordering | 0.0545 | [0.0203, 0.0882] | 159 | 0.0034416 |
| llama_3_1_8b_instruct | framing | 0.2373 | [0.1883, 0.2865] | 160 | 1.6667e-05 |
| llama_3_1_8b_instruct | selection | 0.0312 | [0.0104, 0.0542] | 160 | 0.014721 |
| llama_3_1_8b_instruct | emphasis | 0.1499 | [0.1095, 0.1902] | 160 | 1.6667e-05 |
| llama_3_1_8b_instruct | numerical_loss | 0.0243 | [-0.0031, 0.0526] | 132 | 0.099016 |

Ordering uses the same co-mentioned fact pairs in both outputs, with first mentions across all units and half credit for ties. Comparisons may have different eligible pair sets: do not subtract ordering contrasts as an additive decomposition. Bootstrap: 20,000 paired scenario resamples; two-sided sign flips: 200,000. The ten tests assess goal-minus-new-neutral effects, not before/after changes. Other contrasts are descriptive/sensitivity analyses.

New outputs flagged at the token limit: 0/320. Historical outputs lack token-stop metadata; absence of truncation there is not established.

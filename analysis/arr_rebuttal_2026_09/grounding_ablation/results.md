# Grounding ablation results

Original goal outputs/judgments reused. Only the original neutral grounding sentence was removed. Role/audience differences remain. This is not an objective-only control.

| Model | Metric | Goal minus new neutral | 95% CI | n | BH q (4 primary tests) |
|---|---|---:|---|---:|---:|
| qwen3_8b_no_thinking | ordering | 0.0707 | [0.0267, 0.1165] | 160 | 0.002065 |
| qwen3_8b_no_thinking | framing | 0.1799 | [0.1356, 0.2254] | 160 | 1e-05 |
| llama_3_1_8b_instruct | ordering | 0.0545 | [0.0203, 0.0882] | 159 | 0.002065 |
| llama_3_1_8b_instruct | framing | 0.2373 | [0.1883, 0.2865] | 160 | 1e-05 |

Ordering uses the same co-mentioned fact pairs in both outputs, with first mentions across all units and half credit for ties. Comparisons may have different eligible pair sets: do not subtract ordering contrasts as an additive decomposition. Bootstrap: 20,000 paired scenario resamples; two-sided sign flips: 200,000. Secondary contrasts and dimensions are descriptive/sensitivity analyses.

New outputs flagged at the token limit: 0/320. Historical outputs lack token-stop metadata; absence of truncation there is not established.

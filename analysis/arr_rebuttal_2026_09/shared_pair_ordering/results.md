# Shared-pair Ordering results

Metric: `shared_pairs_all_mentions_ties_half_v1`. Equal scenario weights. Original metric files remain unchanged.

20,000 scenario bootstrap draws; 200,000 two-sided sign flips; main BH family = 12 models × 5 metrics. Complete-both is a separate 12-test sensitivity family.

| Model | n | Paper Ordering → revised | 95% CI | q (60) | Paper Average → revised |
|---|---:|---:|---|---:|---:|
| deepseek_chat | 160 | 0.0965 → 0.1345 | [0.0852, 0.1839] | 0.000014 | 0.062 → 0.070 |
| deepseek_reasoner | 159 | 0.1062 → 0.1705 | [0.1300, 0.2130] | 0.000014 | 0.101 → 0.114 |
| gpt_5_4 | 160 | 0.0479 → 0.0608 | [0.0208, 0.1014] | 0.008069 | 0.053 → 0.056 |
| gpt_5_4_mini | 160 | 0.0375 → 0.0354 | [-0.0028, 0.0747] | 0.129481 | 0.043 → 0.043 |
| llama_3_1_70b_instruct | 160 | 0.0917 → 0.0975 | [0.0606, 0.1355] | 0.000014 | 0.080 → 0.081 |
| llama_3_1_8b_instruct | 159 | 0.0771 → 0.0851 | [0.0503, 0.1209] | 0.000039 | 0.103 → 0.105 |
| qwen3_14b_no_thinking | 160 | 0.0986 → 0.1137 | [0.0729, 0.1556] | 0.000014 | 0.074 → 0.077 |
| qwen3_14b_thinking | 160 | 0.0667 → 0.1283 | [0.0849, 0.1714] | 0.000014 | 0.083 → 0.095 |
| qwen3_32b_no_thinking | 160 | 0.0826 → 0.1281 | [0.0939, 0.1632] | 0.000014 | 0.069 → 0.078 |
| qwen3_32b_thinking | 160 | 0.0451 → 0.0889 | [0.0563, 0.1217] | 0.000014 | 0.059 → 0.068 |
| qwen3_8b_no_thinking | 160 | 0.0465 → 0.0922 | [0.0524, 0.1332] | 0.000048 | 0.065 → 0.074 |
| qwen3_8b_thinking | 160 | 0.0917 → 0.1326 | [0.0927, 0.1734] | 0.000014 | 0.068 → 0.077 |

Excluded model/scenarios: deepseek_reasoner/MED_18, llama_3_1_8b_instruct/MED_14

Displayed Average reproduces the paper table: mean of positive parts of five raw dimension means. This differs from the paper prose (unweighted signed mean). CSV also provides signed averages. Neither is the plotting script’s normalized composite.

Mitigation values are independently paired contrasts; do not subtract them to reconstruct other contrasts because shared-pair sets can differ.

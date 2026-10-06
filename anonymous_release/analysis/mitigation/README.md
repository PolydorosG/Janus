# Prompt-mitigation artifact note

`outputs/mitigation_summary.csv` is the snapshot used for the paper's mitigation
table. Selection, emphasis, ordering, and specificity use the expected available
scenario counts and reproduce from the packaged metric files.

The framing judgments were incomplete when that snapshot was produced. The
reported Llama-3.1-8B framing row uses 54 scenarios, and the reported Qwen3-8B
framing row uses 94 scenarios. The working repository was subsequently updated
with Qwen framing judgments for all 160 scenarios, while the published summary
CSV was not regenerated. Running `experiments/analyze_mitigation.py` against
the currently packaged metric files therefore gives the following Qwen changes:

| Quantity | Paper snapshot (n=94) | Current rerun (n=160) |
|---|---:|---:|
| Guardrail minus original goal, framing | -0.1316 | -0.1194 |
| Guardrail minus neutral, framing | 0.0458 | 0.0654 |

The qualitative conclusion is unchanged: the guardrail reduces Qwen framing
distortion relative to the original goal-conditioned response. The release
retains the paper snapshot under `outputs/` and the latest underlying data under
`metrics/` and `results/`. This distinction should be resolved in the manuscript
or camera-ready artifact rather than hidden in release code.

# Submitted grounding-instruction ablation

Submitted 16 September 2026. This is the author's final requested design and supersedes the earlier proposal to generate both arms or change the neutral role/audience.

For all 160 scenarios, delete exactly this sentence from the original neutral system prompt:

> Use only the provided facts and do not add outside information.

Keep all remaining neutral wording, the user message, and the archived fact order unchanged. Reuse the original goal outputs and their sentence matches/framing judgments without regeneration. There are 160 new neutral outputs per model, 320 new outputs total, and 640 outputs total in the paired analysis.

Models: Qwen3-8B without thinking and Llama-3.1-8B-Instruct. Cached checkpoints are pinned in each manifest. New generations use batch size 4, bfloat16, the original launch-script budget of 768 new tokens, and no temperature override (checkpoint generation defaults). Both cached checkpoints specify sampling with temperature 0.6; Qwen has top-k 20/top-p 0.95 and Llama top-p 0.9. The manuscript's claim that omitting a temperature implies no sampling should be corrected. Historical output records do not include full runtime provenance; matching checkpoint defaults and recorded launch configuration cannot establish historical software/hardware identity. New outputs record their runtime and deterministic per-batch seeds for resumability.

| Stage | Job ID | Dependency |
|---|---:|---|
| New neutral generation, two-task GPU array | 20632142 | None |
| Sentence matching, two-task GPU array | 20632154 | Successful generation array |
| Framing, two-task GPU array | 20632156 | Successful matching array |
| Completeness validation, metrics, statistical analysis | 20632165 | Successful framing array |

Queue verification after submission: generation pending for priority; downstream stages pending on dependencies. The scheduler then estimated generation start at 20 September, 11:00 (scheduler-displayed time); this is provisional, not a guarantee.

Outputs and frozen manifests: `results/arr_grounding_ablation_2026_09/`.

Metrics: `metrics/arr_grounding_ablation_2026_09/<model>/`.

Final analysis: this directory's `results.md`, `summary.csv`, `paired_scenarios.csv`, and `coverage.csv` after the final job succeeds. They do not exist until execution completes.

Logs: `logs/janus_ground_gen-20632142_<0|1>.out` and `.err`; analogous `janus_ground_match-20632154`, `janus_ground_frame-20632156`; final job `janus_ground_analysis-20632165.out` and `.err`. Array index 0 is Qwen, 1 is Llama.

Entry points:

- `experiments/run_grounding_ablation.py`: manifests, provenance, resume checks, reference reuse, and generation.
- `experiments/analyze_grounding_ablation.py`: stage completeness checks and analysis.
- `scripts/slurm_run_grounding_ablation.sh`: GPU stages (`generate`, `match`, `frame`).
- `scripts/slurm_analyze_grounding_ablation.sh`: final CPU stage.
- `experiments/test_grounding_ablation.py`: exact-deletion, pairing, immutable reuse, idempotence, resume rejection, and missing-judgment tests.

The primary contrast is original goal minus new neutral without the grounding instruction. Secondary contrasts are new minus original neutral and original goal minus original neutral. Ordering compares the same co-mentioned fact pairs with first mentions across all units and half credit for ties. Primary inference covers two models × ordering/framing, using 20,000 paired bootstrap draws and 200,000 sign-flip draws with BH correction across four tests. Existing and new goal judgments are not mixed: only the archived goal judgments are reused. Stage validation fails rather than silently analyzing partial judgments.

This tests grounding asymmetry. Role/audience differences remain between the neutral and goal prompts, so it must not be described as an objective-only causal control. The original generic neutral and goal wording are deliberately preserved as requested.

Preliminary `results/arr_matched_prompt_2026_09/*.manifest.json` files were dry-run plans for the superseded fresh-both-arms proposal. No generations or jobs were run for that proposal; it is not an experiment result.

# Paper-to-artifact manifest

## Main benchmark and model results

| Paper component | Released artifact | Reproduction entry point |
|---|---|---|
| 160 items, eight domains | `data/janus.csv` | `python scripts/validate_release.py` |
| Paired prompts and fixed fact order | `janus_payloads.py` | `python janus_payloads.py` |
| Twelve model runs | `results/<model>.jsonl` | `llm_clients.py` |
| Sentence--fact matching | `results/<model>_matched.jsonl` | `judge_fact_units.py` |
| Framing judgments | `results/<model>_framing.jsonl` | `judge_framing.py` |
| Five JANUS dimensions | `metrics/<model>/` | `metrics_pipeline.py` |
| Main model table and domain figures | `metrics/radar_charts/` | `scripts/make_model_radar_charts.py` |
| Paired sign-flip tests, bootstrap intervals and BH correction | `analysis/significance.csv` | `scripts/analyze_significance.py` |
| Shared-pair Ordering definition | `ORDERING.md`, `ordering_metric.py` | `python -m unittest experiments.test_ordering_metric` |
| Absolute inclusion and complete-six Ordering | `analysis/metric_checks/` | `python scripts/analyze_metric_checks.py` |
| Grounding-instruction ablation, all five dimensions | `analysis/grounding/` | `python scripts/reproduce_supplements.py` |

The twelve model stems are `gpt_5_4`, `gpt_5_4_mini`, `deepseek_chat`,
`deepseek_reasoner`, `llama_3_1_8b_instruct`, `llama_3_1_70b_instruct`, and the
8B/14B/32B Qwen3 thinking and no-thinking variants.

## Supplemental analyses

### Fact-valence sanity check

The released Llama-3.1-70B classifications are in
`analysis/valence/meta_llama_llama_3_1_70b_instruct.jsonl`. Recompute the
eight-scenario sensitivity analysis with:

```bash
PYTHONPATH=experiments python experiments/analyze_valence_sensitivity.py \
  analysis/valence/meta_llama_llama_3_1_70b_instruct.jsonl \
  --metrics-root metrics \
  --output-dir reproduced_analysis/valence_sensitivity
```

### Cross-family framing judge

```bash
python experiments/analyze_framing_judges.py \
  --original-dir results \
  --alternate-dir analysis/judge_robustness/alternate_judgments \
  --alternate-tag framing_llama70 \
  --output-dir reproduced_analysis/judge_robustness
```

### Prompt-level mitigation

The two three-condition generation files and judge outputs are in
`analysis/mitigation/results/`. Their metrics are in
`analysis/mitigation/metrics/`. Recompute the summary from the latest packaged
judgments with:

```bash
python experiments/analyze_mitigation.py \
  --run Qwen3-8B \
    analysis/mitigation/metrics/qwen3_8b_no_thinking_self_awareness_direct/core_metrics.csv \
    analysis/mitigation/metrics/qwen3_8b_no_thinking_self_awareness/core_metrics.csv \
  --run Llama-3.1-8B \
    analysis/mitigation/metrics/llama_3_1_8b_instruct_self_awareness_direct/core_metrics.csv \
    analysis/mitigation/metrics/llama_3_1_8b_instruct_self_awareness/core_metrics.csv \
  --output-dir reproduced_analysis/mitigation
```

The summary uses the current packaged judgments and the shared-pair Ordering
metric. Qwen framing has 160 scenarios; Llama framing has 54. See
`analysis/mitigation/README.md`. `python scripts/reproduce_supplements.py`
also rebuilds the underlying mitigation metrics from the raw files.

### Grounding and metric checks

`analysis/grounding/results/` contains 160 new neutral responses per model and
the original goal responses, with sentence matches and framing judgments.
Only the sentence restricting neutral responses to the provided facts was
removed. `experiments/analyze_grounding_ablation.py` checks prompt differences,
fact order, reused goal text, match alignment, and judgment completeness.
The ten goal-minus-new-neutral tests (five dimensions by two models) share a
BH correction. This is not an objective-only control: roles and audiences remain.

`analysis/metric_checks/` reports per-model inclusion of both fact classes and
Ordering restricted to scenarios with all six facts in both responses. The
latter uses a separate twelve-test BH family.

The full supplemental reproduction command is:

```bash
python scripts/reproduce_supplements.py
```

## Deliberate exclusions

The package excludes `.env`, provider batch request/response envelopes,
reasoning traces, lock files, logs, Slurm scripts and state, caches, paper
source, author working documents, draft responses, and reviewer material. None
is required to recompute the released benchmark measurements from the normalized
JSONL results.

# JANUS anonymous code and data release

This directory is a self-contained artifact for the paper *JANUS: Evaluating
Goal-Conditioned Distortion in Factually Grounded LLM Communication*. It
contains the 160-scenario benchmark, the twelve primary generation runs,
sentence--fact matches, framing judgments, derived metrics, and the code used
for the paper's main and supplemental analyses.

The release intentionally excludes author identities, institutional details,
API credentials, local paths, scheduler files, logs, caches, batch-provider
request files, reasoning traces, drafts, and reviewer correspondence.

## Provenance notes

The twelve primary pipelines reproduce from the packaged files. All Ordering
results use the shared-pair, first-mention, half-credit-for-ties definition in
`ORDERING.md`. There is no alternative Ordering implementation or historical
Ordering snapshot in this release.

Mitigation summaries use the packaged judgments: all 160 Qwen scenarios and
54 Llama framing scenarios. See `analysis/mitigation/README.md` for scope.

## Contents

- `data/janus.csv`: 160 scenarios across eight domains, with three favorable
  and three adverse facts per scenario.
- `results/`: the twelve primary runs. For each model, `<model>.jsonl` contains
  the paired generations, `<model>_matched.jsonl` contains sentence--fact
  matches, and `<model>_framing.jsonl` contains framing judgments.
- `metrics/`: per-model fact-, output-, scenario-, and category-level metrics,
  plus the aggregate tables and figures in `metrics/radar_charts/`.
- `janus_payloads.py` and `llm_clients.py`: paired prompt construction and
  generation through local Hugging Face models or supported API providers.
- `judge_fact_units.py`, `judge_framing.py`, and `judge_prompts.py`: the two
  model-judge stages and their exact prompts.
- `metrics_pipeline.py`: deterministic computation of selection, emphasis,
  ordering, specificity, and framing metrics.
- `ordering_metric.py` and `ORDERING.md`: the authoritative paired Ordering
  implementation and mathematical definition.
- `scripts/`: one-command reproduction, significance testing, plotting, and
  release-integrity checks.
- `analysis/`: paper supplements for cross-family judge robustness, prompt
  mitigation, and fact valence sensitivity.

See `MANIFEST.md` for a paper-to-artifact map and `DATA_CARD.md` for the dataset
schema and intended use.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python -m pip install -r requirements.txt
```

For offline metric/figure reproduction only, install
`requirements-reproduce.txt` instead; model inference libraries are not needed.

Local generation and judge runs download model checkpoints unless they are
already cached. API runs read credentials from environment variables:
`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or `ANTHROPIC_API_KEY`. Never commit a
`.env` file.

## Reproduce the released metrics and figures

The following command starts from the released raw generations and judge
outputs. It does not call an external model or API.

```bash
python scripts/reproduce_main_results.py
```

This writes a fresh copy to `reproduced_metrics/`, leaving the released
artifacts unchanged. To reproduce the paper's paired sign-flip tests:

```bash
python scripts/analyze_significance.py
```

The paper uses 200,000 Monte Carlo draws. For a quick smoke test, pass
`--draws 1000`.
The default seed is 160; 95% intervals are percentile intervals from 20,000
paired-scenario bootstrap resamples. The 60 main tests share one BH correction.

## Run a new benchmark

This example runs both conditions for Qwen3-8B without thinking:

```bash
python llm_clients.py \
  --provider qwen_local \
  --model Qwen/Qwen3-8B \
  --qwen-thinking false \
  --dtype bfloat16 \
  --batch-size 4 \
  --max-tokens 768 \
  --output results/my_model.jsonl
```

Then evaluate and aggregate the run:

```bash
python judge_fact_units.py results/my_model.jsonl \
  --output results/my_model_matched.jsonl

python judge_framing.py \
  results/my_model.jsonl \
  results/my_model_matched.jsonl \
  --output results/my_model_framing.jsonl

python metrics_pipeline.py \
  results/my_model.jsonl \
  results/my_model_matched.jsonl \
  --framing-jsonl results/my_model_framing.jsonl \
  --output-dir metrics/my_model
```

All stages are resumable by default. The same fact-shuffle seed is keyed by
scenario, so neutral and goal-conditioned prompts receive the same fact order.

## Supplemental analyses

`MANIFEST.md` gives exact commands. In brief, the release includes:

- the 960-fact valence classification and eight-scenario sensitivity analysis;
- Llama-3.1-70B alternate framing judgments for the cross-family robustness
  comparison;
- two prompt-guardrail runs and their direct/residual metrics.
- absolute favorable/adverse inclusion and the all-six-facts Ordering subset;
- the two-model grounding ablation, including normalized generations and
  judgments, all five dimensions, and a ten-test BH correction.

```bash
python scripts/reproduce_supplements.py
```

This command refreshes the packaged supplemental metrics and summaries in
`analysis/`. It makes no model/API calls and does not modify raw generations.

## Integrity check

Before uploading or linking the artifact, run:

```bash
python scripts/validate_release.py
```

The check verifies dataset balance, expected row counts, required artifacts,
paired Ordering consistency, and the absence of common private or
machine-specific material. Run the metric tests with:

```bash
python -m unittest experiments.test_ordering_metric experiments.test_release_results
```

Upload only this directory, not its parent repository or Git history. Hosting
account names and upload metadata must also be anonymous; local checks cannot
verify those. See `RELEASE_CHECK.md` for validation scope and remaining decisions.

## License

No license was present in the source repository. The artifact owner should add
the intended code and dataset license before public distribution. Until then,
normal copyright restrictions apply.

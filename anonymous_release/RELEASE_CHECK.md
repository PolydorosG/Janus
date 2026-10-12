# Anonymous release validation

## Status

The packaged data and deterministic analyses passed local validation. The
Ordering definition, code, CSVs, tables, figures, and statistical analyses use
only the shared-pair first-mention metric with half credit for same-sentence
ties. Superseded Ordering functions and output columns have been removed;
old result snapshots are not included.

## Checks performed

- Recomputed all 12 primary runs from packaged raw responses, sentence matches,
  and framing judgments, then regenerated all data plots and both LaTeX tables.
- Verified all 3,840 primary responses and their matching scenario/condition
  keys, six source facts, sentence alignment, and complete unique framing keys.
- Checked each of the 1,920 primary scenario Ordering values against the shared
  pair implementation. Empty comparisons remain undefined.
- Compared against the revised manuscript analysis: all 60 p-values and BH
  q-values agree exactly; means and interval endpoints agree within 2e-13.
  All 12 Ordering means are positive and 11 are significant after BH across 60
  tests. The other four primary metric columns are unchanged.
- Rebuilt mitigation metrics and summaries, using current packaged judgments
  rather than retaining a stale snapshot. Llama mitigation framing is available
  for only 54 scenarios; this remains explicitly documented.
- Reproduced the grounding ablation from normalized raw artifacts, checking
  prompt changes, fixed fact order, reused goal responses, sentence alignment,
  and all 5,602 framing judgments. Reports all five dimensions with BH across
  ten tests.
- Recomputed absolute inclusion and all-six-facts Ordering robustness: 1,739
  complete paired scenarios across models; 12 positive effects, 11 significant
  under the separate twelve-test BH family.
- Corrected numerical tie handling in valence sensitivity rankings. The
  five-dimension Spearman range is 0.958–0.993.
- Eleven unit/regression tests pass; all Python sources compile.
- Scanned text artifacts for email addresses, machine-specific paths, and
  common credential patterns; checked for caches, private directories, and
  symlinks. No violations found. PNGs contain software/DPI metadata, not author
  metadata. No raw generation or primary judge data were changed.

## Recheck before upload

From this directory:

```bash
export PYTHONDONTWRITEBYTECODE=1
python -m unittest experiments.test_ordering_metric experiments.test_release_results
python scripts/validate_release.py
```

To regenerate primary results in a separate directory:

```bash
python scripts/reproduce_main_results.py
python scripts/analyze_significance.py
```

To refresh the supplemental derived artifacts in place:

```bash
python scripts/reproduce_supplements.py
```

These checks ran with Python 3.9.13, NumPy 1.26.4, Matplotlib 3.5.2 and NLTK
3.9.1. Python 3.10 or newer is recommended for new installations. A fresh
environment installation and fresh GPU/API generations were not tested;
deterministic reproduction uses no model calls or downloads.

## Owner decisions and external checks

- No code/data license has been selected. Add your intended licenses before
  distributing this as a reusable public release; no license was invented here.
- Upload this directory alone. Do not include parent-repository history or
  working files. Use an anonymous hosting account/link and check its visible
  owner information and upload metadata. Local scans cannot validate hosting
  anonymity or guarantee the absence of every possible identifying clue.
- Human annotation-level records are not packaged; this release does not claim
  to reproduce human-validation statistics from individual annotations.

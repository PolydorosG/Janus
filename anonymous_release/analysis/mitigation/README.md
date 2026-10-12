# Prompt-mitigation artifact note

The metrics and summaries are recomputed from the packaged generations and
judgments. Ordering uses the shared-pair definition in `ORDERING.md` for
both direct (guardrail minus goal) and residual (guardrail minus neutral)
comparisons. Each comparison constructs its own shared pair set; do not obtain
one Ordering contrast by subtracting two others.

Qwen framing has judgments for all 160 scenarios. Llama framing has judgments
for 54 scenarios; this limitation is retained explicitly rather than imputing
the missing judgments. Non-framing metrics use their own eligible scenarios.

Current residual framing effects are 0.0654 for Qwen and 0.0963 for Llama.
The Qwen direct framing effect is -0.1194. These are calculated from the
current data, not an earlier partial-judgment snapshot.

Direct Ordering effects are -0.1155 (Qwen, n=160) and -0.0423 (Llama, n=159).
Residual Ordering effects are -0.0266 (Qwen, n=160) and 0.0431 (Llama, n=160).
Direct 95% intervals use 20,000 scenario bootstrap resamples and seed 160.

Run `python scripts/reproduce_supplements.py` from the release root to
rebuild the metrics and summaries without new generations or judgments.

# Completed round-two manuscript revision — 12 October 2026

Authority: the posted response in `../rebuttal_draft_compressed.md`, the three reviews in `reviews/reviews.md`, and the current user-supplied manuscript/images. The posted rebuttal and previous-round revision letter were not edited.

## Manuscript deliverables

- `paper/acl_latex.tex`: shared-pair ordering definition; corrected paired-effect interpretation; refreshed main results, category discussion, significance explanation, mitigation values and intervals, human-validation association, and qualitative-example ordering; links to new appendices; clarified role/objective and grounding scope; numerical specificity qualification; absolute-coverage interpretation. All five image includes now resolve to the supplied `paper/images/` tree.
- `paper/appendices/reviewer_analyses.tex`: absolute inclusion, shared-pair/all-six ordering, complete five-dimensional grounding ablation, numerical specificity, and aggregation/figure interpretation.
- `paper/tables/model_metrics.tex`: all 12 revised ordering means, averages, maxima/runner-up emphasis, and recomputed BH shading across 60 tests.
- `paper/tables/absolute_coverage.tex`: full per-model favorable/adverse inclusion and complete-output counts in both conditions.
- `paper/tables/ordering_robustness.tex`: main and all-six subset means, individual percentile bootstrap intervals, and per-model sample sizes.
- `paper/tables/grounding_ablation.tex`: all ten model/dimension comparisons, original and sentence-removed effects, intervals, eligible counts, and ten-test BH q-values.
- `paper/images/category_radar.png`, `image.png`, `all_radars.png`: revised values using the original plotting functions and visual settings. The last image is the supplied auxiliary grid and remains unused in the main TeX, as before. The conceptual overview, teaser, and logo are unchanged. The overview caption now identifies schematic bars and clarifies the shared-pair and numerical-loss interpretations.
- `paper/revisions/round2_revisions.tex`: a new revision letter following the round-one template, mapping every current reviewer concern to implemented changes. `round1_revisions.tex` is preserved.

## Verified numerical changes

- Main ordering: 12/12 positive, 11/12 significant (60-test BH); GPT-5.4 Mini q=0.129481. Two excluded model/scenario comparisons, never imputed as zero.
- Complete-six subset: 12/12 positive, 11/12 significant (separate 12-test BH); 132–154 scenarios per model.
- Grounding ablation: all five dimensions reported with one ten-test family. Eight effects significant; numerical loss non-significant for both models. Ordering Qwen 0.0922→0.0707 and Llama 0.0851→0.0545, both q=0.00344165. This tests effects after removing the sentence, not before/after changes or equivalence.
- Main table positive-part Average: DeepSeek Thinking 0.114, Llama 8B 0.105. The earlier manuscript called the aggregation an unweighted mean without explaining clipping; the revision preserves the numerical convention and explains it explicitly.
- Mitigation: Qwen residual ordering −0.0266, Llama 0.0431. Independent direct contrasts −0.1155 and −0.0423. Their pair sets differ, so contrasts are not reconstructed by subtracting aggregate table values.
- Ambiguity sensitivity: ordering Spearman 0.9580; five-dimension range 0.952–0.993.
- Human validation: same 50 pairs and 100 judgments; correlation recomputed from revised scenario scores is 0.508494, two-sided 200,000-draw pair-level permutation p=0.000179999. Original 0.502800 reproduced independently. High/low goal-misleading votes remain 28/50 and 7/50. The actual historical selection code used quartile-stratified sampling, not the absolute top/bottom 25; prose corrected accordingly.
- Fixed qualitative examples: Qwen32 Thinking/BUS_17 ordering 0.2222; Qwen32 non-thinking/WOR_10 ordering 0.0000; both share all nine pairs.

## Figure preservation audit

`figure_audit.json` records pixel checks. Rendering the original numerical values with the existing plotting functions reproduces all three supplied data images exactly (RGB pixels, excluding file metadata). Rendering the revised dumbbell changes only pixel columns 640–752, inside the Ordering lane; every non-ordering pixel is unchanged. The category/radar outlines necessarily move where ordering affects composites or their established normalization; color, layout, label, model-order, size, and scaling rules are preserved.

`original_figures/` contains recoverable copies of the three supplied images before replacement. `legacy_*_reproduction.png` files are audit reproductions, not manuscript figures. The figures' short Specificity labels are deliberately retained to respect the requested visual freeze; their numerical-only meaning is stated in the manuscript/captions.

## Reproduction

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/analyze_arr_review_robustness.py
PYTHONDONTWRITEBYTECODE=1 python experiments/analyze_shared_pair_ordering.py
PYTHONDONTWRITEBYTECODE=1 python experiments/analyze_grounding_ablation.py
PYTHONDONTWRITEBYTECODE=1 python experiments/analyze_valence_sensitivity.py analysis/valence/meta_llama_llama_3_1_70b_instruct.jsonl --metrics-root analysis/arr_rebuttal_2026_09/shared_pair_ordering/metrics --output-dir analysis/arr_rebuttal_2026_09/shared_pair_ordering/valence_sensitivity
PYTHONDONTWRITEBYTECODE=1 python experiments/prepare_round2_manuscript.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest experiments.test_ordering_metric experiments.test_grounding_ablation experiments.test_round2_manuscript
```

The manuscript generator preserves the other four metrics' historical values and plotting transformations. `human_rescored_pairs.csv` and `human_association.json` audit the fixed human sample; `category_composite_scores.csv` contains the regenerated category values. Main/complete/mitigation inference is in `../shared_pair_ordering/`; full ablation inference is in `../grounding_ablation/`.

Validation: 17 unit/source/data tests passed, including resolution of all TeX inputs, figure paths and cross-references, balanced LaTeX environments, table consistency with recalculated means, ten-test correction coverage, fixed annotation counts, and pixel-level visual preservation. Scientific figures were inspected visually. This is source/data validation, not a claim of a compiled-PDF layout check.

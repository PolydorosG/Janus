# Rebuttal strategy — current ARR reviews

Execution update: the author selected a narrower grounding ablation, deleting only the original neutral grounding sentence and reusing original goal outputs. It is submitted; see `grounding_ablation/RUN.md`. The role-matched objective-only experiment discussed below remains an optional, unrun alternative. The submitted ablation cannot remove the role/audience confound.

The strongest response is to accept the measurement criticisms, show that the main ordering pattern survives a more defensible analysis, and directly test the prompt confound. The reviewers already credit the motivation, breadth, writing, and validation. We should focus their remaining uncertainty on concrete evidence rather than repeat those strengths at length.

The 3/3/3.5 scores are supplied by the author; the local review transcription does not map scores to reviewer IDs. Do not infer which reviewer gave which score. The July drafts concern a different review round and should not be reused as the response to these reviewers.

## Priorities and current evidence

| Concern | Reviewers | Assessment | Best response | Status |
|---|---|---|---|---|
| Omission can raise Ordering | All three | Correct | Compare only co-mentioned fact pairs, then the same pairs in both conditions; handle ties | Completed on all 12 models |
| Near-zero Selection does not prove retention | K6Zq | Correct | Absolute favorable/adverse coverage and all-six retention, per model and overall | Completed |
| Prompt differences bundle grounding, role/audience, objective | Vvyc, GTaK | Correct; largest remaining threat to causal interpretation | Matched objective-only contrast, with no explicit grounding restriction in either arm and identical role/audience | Exact protocol in `matched_prompt_plan.md`; no generation run yet |
| Specificity is only numerical and not necessarily goal-favoring | GTaK | Correct | Rename, state numerical/semantic limitations, separate from directional interpretation | Drafted |
| Capitalization | Vvyc | Easy presentation fix | Consistent lowercase/italics in prose | Drafted commitment |

## What we can already say confidently

The new script reproduces all five published mean metrics for all 12 model configurations from the cached metric files, and independently checks cached fact matches against the original matched JSONL. It verifies that the stored scenario and ordered evidence agree within every primary prompt pair. All 3,840 primary outputs have complete cached framing judgments. This does not revalidate the judges themselves, but establishes that the offline analyses use the intended primary artifacts.

Absolute favorable/adverse coverage is 99.306%/99.601% in the neutral condition and 98.854%/98.524% in the goal condition. The all-six rate drops from 96.77% to 92.86%, so avoid saying that inclusion is unchanged. Llama-3.1-8B's adverse coverage falls by 3.33 percentage points, the largest decrease. The defensible conclusion is high observed source-fact coverage with a modest aggregate decline, not proof of coverage from a near-zero imbalance.

The cleanest ordering rebuttal compares the *same fact pairs* in both outputs. It uses all first matched mentions and half credit for a tie, rather than treating mixed-polarity units as absent. Mean shifts are positive for all 12 models, and 11 pass the 12-test BH correction. GPT-5.4 Mini has a positive but uncertain mean. Report this exception plainly. The all-six-in-both subset gives the same qualitative direction for all models, with 132–154 scenarios per model. This subset is a robustness analysis conditioned on outputs, not an unbiased estimate for every possible response.

These findings are stronger evidence against an omission-only explanation than merely observing high average coverage. They still do not identify which part of the prompt intervention caused the ordering change.

The narrower denominator-only check, retaining the old first-eligible-unit convention, also gives positive shifts for all models, with all 12 passing BH correction. Do not lead with that stronger significance count: it retains the mixed-unit issue. Lead with the more defensible common-pair analysis and disclose the uncertain Mini result.

## The experiment most likely to change the assessment

Run the matched-prompt contrast before submitting the final responses. Time is available, and this concern cannot be resolved by changing the wording of the old templates without generating new outputs. Following the author's preference, omit the explicit grounding restriction from both arms. Hold the institutional role and exact audience wording constant and add only the objective sentence. This preserves the original goal-conditioned prompt and avoids adding a new intervention to it. Simply removing the restriction from the original generic neutral prompt would still leave the role/audience confound.

Start with Qwen3-8B and Llama-3.1-8B across all 160 scenarios: they cover two families and already have cross-family judge checks and mitigation results. This is a manageable control experiment, not evidence about every model. If it succeeds and resources permit, extend to a prespecified larger or closed-source configuration; do not select subsequent models based on which result looks best. Keep old results labeled as the original bundled intervention.

An optional third arm removes only the grounding restriction from the original generic neutral prompt. It connects the original comparison to the new matched contrast: original neutral → unrestricted generic neutral → unrestricted institutional-role baseline → original goal. This identifies where changes arise along that sequence without adding any constraint to the goal prompt. It is not a full factorial design or a unique decomposition in the presence of interactions.

If the matched effect weakens substantially, that is scientifically useful: revise the conclusion to institutional prompt sensitivity and report the attenuation. If it disappears, retain the benchmark contribution but abandon the objective-only empirical claim. No amount of re-emphasizing the existing human validation removes this confound.

## Specificity: make the concession cleanly

Rename it *numerical retention*, or use *numerical-detail loss* where higher values mean more loss. Positive loss is nondirectional with respect to the institutional goal. Remove it from any interpretation of the five-way average as a scalar amount of goal-favoring distortion; retain the five-dimensional profile and label any aggregate descriptive and heterogeneous. Avoid introducing a new signed numerical score during the rebuttal unless it is independently justified.

The current extractor finds numbers in 132/160 scenarios. It uses normalized values across the whole response and does not verify that a number remains attached to the correct fact, unit, or qualifier. A diagnostic fact-matched polarity split is in `numeric_retention_by_polarity.csv`: favorable retention averages 94.39% → 93.42%, adverse retention 96.05% → 93.26%. These are occurrence-weighted within each polarity, then averaged over models, and are not directly the published scenario-averaged score. Do not present this optional diagnostic as a validated goal-direction metric. A retained number can also occur in a sentence matched to several facts.

## Internal corrections discovered during the audit

These matter for the eventual manuscript and release. They are not a reason to burden every reviewer response with unrelated details.

1. **Mixed-polarity handling is missing from the mathematical account.** `metrics_pipeline.py` records inclusion from all matched units but skips mixed-polarity units for emphasis token allocation and first positions. The paper describes splitting tokens across matched facts without this exclusion. `scripts/make_model_radar_charts.py` then replaces the pipeline's mean-position ordering with the pairwise score used in the paper. The revised methodology/code must agree. The new ordering analysis explicitly includes all matched units and handles ties; it does not silently modify the main pipeline.

2. **Fixed evidence is not guaranteed output factuality.** The abstract, introduction, conclusions, and limitations use stronger “isolates” language than the design supports. The H03 example itself acknowledges an unsupported premiums claim. Replace the guarantee with “holds supplied evidence constant and measures its treatment”; separate unsupported additions from presentation. Grounding instructions still do not themselves certify factuality.

3. **Human sample description needs correction.** The 100 individual ratings reproduce the paper's 40/50 vs. 25/50 action-induction and 28/50 vs. 7/50 misleadingness counts. However, `analysis/persuasiveness_v2/sampling_manifest.json` and the sampler indicate 25 examples sampled from each score quartile, excluding 50 prior pilot pairs—not literally the 25 highest and 25 lowest examples. The stored annotation instruction files also contain differing protocol versions. Describe the actual form shown to annotators and the actual sampling. Do not conflate the separate AI pilot/assistant annotations with the two human response rows.

4. **Mitigation coverage and provenance need repair.** The current Qwen mitigation output metrics contain all 160 framing pairs and reproduce 0.0654. The Llama mitigation metrics contain framing scores for only 54 paired scenarios and have missing sentence–fact judgments; 0.0963 is an available-case mean, not demonstrated full-160 framing performance. Its neutral/goal/mitigated conditions have 875/944/920 missing pair judgments in that cache. The mitigation runner copies the original baseline outputs, contrary to prose suggesting independently regenerated baselines. Report the provenance and complete judging before making full-sample mitigation claims. These issues do not affect the complete primary-run framing judgments or the new primary ordering checks.

5. **Terminology for ordering.** The paper's “relative to the provided order” is not the formula shown: the pairwise score measures output order. Identical input order cancels in the paired comparison; it is not explicitly in the absolute score. State that distinction. Same-unit co-mentions supply no within-unit ordering information.

## Manuscript revision map

| Location | Revision |
|---|---|
| Abstract, Introduction, contributions, conclusions | Replace objective-isolation and guaranteed-factuality claims; state the actual intervention and scope |
| Paired Benchmark Design | Separate original prompts from the new matched control; give exact templates and what changes |
| Evaluation: Ordering | Define present/common pair sets, first matched units, ties, and undefined cases; disclose mixed-unit treatment |
| Evaluation: Selection; Main Results | Add absolute class coverage and all-six rates; revise inference from imbalance |
| Evaluation: Specificity; dimension table; paired distortion paragraph | Rename; report 132 eligible scenarios; remove universal goal-direction claim |
| Main results and appendix | Add model-level robustness means, intervals, counts; label original vs corrected metrics |
| Human validation appendix | Correct sampling/protocol description without changing authentic human judgments |
| Mitigation appendix | Correct baseline provenance and framing denominators; complete missing judgments before full-sample claims |
| Throughout | Consistent dimension typography |

Changing ordering or the aggregate score also affects rankings, plots, and any human-validation association based on that aggregate. Retain the historical score used to select the human sample and label it; do not silently relabel old samples as selected by a revised score. Recompute downstream quantities deliberately if adopting the revised metric in the main paper.

## Response sequencing and tone

Give each reviewer a self-contained response. K6Zq's two concerns can already be answered with completed evidence. Vvyc and GTaK need the matched-prompt result for the strongest reply. Open with agreement on the valid point, give the relevant numbers, then state exactly what will change. Do not say that differencing automatically removes prompt or judge confounds. Do not claim that favorable ordering alone proves recipient harm, or that every nonzero presentation change is deception.

The reviewer-facing draft deliberately contains no external links. Current [ARR author guidance](https://aclrollingreview.org/authors#step-2-respond-to-reviews) permits small new experiments directly addressing reviewer questions, requires text-only responses without external links, and favors focused discussion. Check the cycle's OpenReview instructions before posting. Keep the scientific evidence in the response itself rather than asking reviewers to inspect a repository. Aim for one substantive response per reviewer and one follow-up if needed, without pressing for score increases.

## Files and reproduction

- `rebuttal_draft.md`: complete responses using only currently verified evidence, plus instructions for replacing the prompt-control paragraph after the new experiment.
- `matched_prompt_plan.md`: exact design, controlled strings, contrasts, and reporting requirements.
- `verified_results.md`: readable coverage and ordering tables.
- `ordering_by_model.csv`: all six ordering variants, paired-bootstrap intervals, sign-flip p-values, and within-variant BH q-values.
- `scenario_checks.csv`: 1,920 model/scenario records, including eligibility and complete-case flags.
- `artifact_audit.csv`: prompt, cached-match, main-metric, and primary-framing completeness checks.
- `numeric_retention_by_polarity.csv`: optional descriptive numerical diagnostic.

Run `python experiments/analyze_arr_review_robustness.py` from the repository root. It uses NumPy, reads existing artifacts, and writes only this analysis directory. It performs no model inference, network calls, or edits to the paper, original results, or existing analysis files.

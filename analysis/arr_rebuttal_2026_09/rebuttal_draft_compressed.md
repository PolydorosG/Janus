# ARR rebuttal draft

Author note: the responses below use completed analyses with the shared-pair Ordering metric. The local paper definition has been updated; remaining numerical tables, figures, and narrative changes are listed in `shared_pair_ordering/paper_update_checklist.md`.

## Response to Vvyc

We thank reviewer Vvyc for the review.

W1: Ordering and omission. We agree the original ordering metric could be affected by missing facts. We therefore recalculated it using only favorable-adverse pairs whose two facts appear in both neutral and goal-conditioned responses. Missing pairs are excluded, so omission cannot itself count as reordering. We use each fact’s first mention, assign 0.5 to same-sentence ties, compute the within-scenario goal-minus-neutral difference, and average scenarios equally.

The revised ordering effect remains positive for all 12 models (0.0354–0.1705) and significant for 11 after adjustment across all 60 model-metric tests; previously 10 were significant. GPT-5.4 Mini is positive but not significant (0.0354; 95% interval [−0.0028, 0.0747]). Restricting further to scenarios where all six facts appear in both responses gives the same qualitative result: all 12 means are positive (0.0241–0.1734), with 11 significant. We will update the metric and report full per-model results and this robustness check. The full formula and results are reserved for the final manuscript due to the character limit of this ARR cycle.

W2: Prompt imbalance. We removed only the neutral grounding sentence and regenerated all 160 neutral responses for Qwen3-8B and Llama-3.1-8B, reusing the original goal responses. Results remain similar: Selection 0.0146→0.0229 / 0.03125→0.03125; Framing 0.1848→0.1799 / 0.2342→0.2373; revised Ordering 0.0922→0.0707 / 0.0851→0.0545; Specificity −0.0046→−0.0055 / 0.0269→0.0243; Emphasis 0.0788→0.0736 / 0.1457→0.1499. After BH adjustment across ten tests, selection, emphasis, ordering, and framing are significant for both models; specificity is not. Thus, for both tested models, removing this instruction does not materially change the overall pattern. We will add this experiment in the revised manuscript.

C1. We will use consistent lowercase, italicized dimension names.


## Response to K6Zq

We would like to thank reviewer K6Zq for their review. 

**W1: Interpreting *selection*.** We agree it would also be helpful to report how many favorable and adverse facts are actually included rather than the imbalance on its own. *Selection* measures a change in the balance between the two, so a small score alone cannot establish that the evidence is retained. We checked fact inclusion across all 3,840 responses using the existing sentence-fact matches:

| Supplied facts included | Neutral | Goal-conditioned |
|---|---:|---:|
| Favorable facts | 99.31% | 98.85% |
| Adverse facts | 99.60% | 98.52% |
| Responses including all six facts | 96.77% (1,858/1,920) | 92.86% (1,783/1,920) |

Both types of fact are retained at high rates, although inclusion falls under goal conditioning, more for adverse facts (1.08 percentage points) than favorable facts (0.45 points). The aggregate also hides model differences: goal-conditioned inclusion ranges from 98.12-99.79% for favorable facts and 95.83-99.58% for adverse facts. Llama-3.1-8B has the largest adverse-fact decline, from 99.17% to 95.83%. These results support high overall retention, not complete retention or an absence of selective omission. We will revise the claims based on *selection* alone and report the full per-model results in the appendix of the revised version.

**W2: Ordering and missing facts.** We agree that the paper’s *ordering* formula was affected by fact omissions, and have since updated it, as described in our **W1 response to reviewer Vvyc**, where we provide the results.

As an additional check, restricting to scenarios where all six facts appear in both responses also gives positive mean effects for every model (132-154 scenarios per model). This supports an ordering change even where neither response omits any fact. We will include this check in the appendix (see our **W1 response to reviewer Vvyc** for the exact results).

## Response to GTaK
We thank the reviewer for the constructive review.

W1: Grounding, roles, and audiences. We agree the prompts differ in more than the objective sentence, partly by design. The grounding asymmetry is addressed in Vvyc W2: removing only the neutral grounding instruction and regenerating all 160 neutral responses for Qwen3-8B and Llama-3.1-8B leaves the main pattern intact.

The audience is present in both conditions via the user message (Figure 10). The goal-conditioned system message restates the same audience context. We omit the role from the neutral condition because roles can themselves imply a goal: e.g., “enrollment specialist” may suggest an interest in increasing enrollment without an explicit objective. Including it would risk goal-conditioning the baseline and test a different research question.

Our paired design is designed to measure institutional goal conditioning (i.e. institutional role + objective) relative to neutral decision support, not the incremental effect of the objective sentence alone.

W2.1: Ordering and omission. We revised ordering to condition on pairs present in both responses. Effects remain positive for all 12 models, including when all six facts are retained; see Vvyc W1.

W2.2: Specificity. We agree the metric captures retention of numbers rather than all forms of qualification and will rename it accordingly. We also agree that numerical loss is not inherently goal-favoring. Its purpose is to measure whether this information is faithfully preserved. Losing a number can alter interpretation in either direction: omitting the size of a cost may downplay a risk, while replacing “returns increased by 5%” with “returns increased dramatically” may exaggerate a benefit. In both cases, the recipient loses information needed to judge the strength of the claim. We therefore treat numerical specificity as information preservation, and address directional goal-favoring presentation through framing.


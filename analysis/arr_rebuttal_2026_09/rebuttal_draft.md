# ARR rebuttal draft

Author note: the responses below use completed analyses with the shared-pair Ordering metric. The local paper definition has been updated; remaining numerical tables, figures, and narrative changes are listed in `shared_pair_ordering/paper_update_checklist.md`.

## Response to Vvyc

We thank reviewer Vvyc for the careful and constructive review and for recognizing the novelty and rigor of our work. Below, we address both weaknesses in the order they appear in the review.

**W1: Ordering and omission.** We agree that our current definition of *ordering* may be affected by missing facts. After further review we also noticed edge cases when a single sentence included more than one fact. We therefore recalculated the effect by comparing only favorable–adverse pairs whose two facts appear in both the neutral and goal-conditioned responses. A missing pair is excluded from both sides, so dropping a fact does not itself count as reordering the retained facts. We use each fact's first mention across all matched sentences, including sentences containing both favorable and adverse facts, and give half credit when both facts first appear in the same sentence.

Let $\mathcal C(x,M)$ contain these shared pairs for scenario $x$ and model $M$, and let $o_y(f)$ be the first sentence mentioning fact $f$. The revised definition is:

$$
s_y(f^+,f^-)=
\begin{cases}
1 & \text{if }o_y(f^+)<o_y(f^-),\\
\tfrac12 & \text{if }o_y(f^+)=o_y(f^-),\\
0 & \text{if }o_y(f^-)<o_y(f^+).
\end{cases}
$$

$$
D_{\mathrm{ord}}(x,M)=
\frac{1}{|\mathcal C(x,M)|}
\sum_{(f^+,f^-)\in\mathcal C(x,M)}
\left[s_{y^{G}}(f^+,f^-)-s_{y^{N}}(f^+,f^-)\right].
$$

We calculate this difference for each scenario, then average scenarios equally for each model. Scenarios without a shared pair are excluded (one each for DeepSeek v4 Thinking and Llama-3.1-8B). 

The *ordering* effect remains positive for all 12 models (0.0354–0.1705) and statistically significant for 11 using the paper's testing procedure, including adjustment across all 60 model–metric tests. Under the original metric 10 models were significant. GPT-5.4 Mini is positive but not significant: 0.0354, with a 95% interval of [−0.0028, 0.0747]. The average effect increases for 11 models and decreases slightly for one. Thus, effect sizes change, but the ordering pattern persists when comparing the same retained evidence. 

As a further check, we restricted the analysis to scenarios where *all six* facts appear in both responses. Mean ordering effects remain positive for all 12 models, ranging from 0.0241 to 0.1734, and 11 remain significant after adjustment across the 12 model tests. GPT-5.4 Mini is again the exception: 0.0241, with a 95% bootstrap confidence interval of [−0.0157, 0.0636]. Thus, the favorable-first shift persists even when neither response omits any fact. We will report the full per-model results in the appendix.. We will update the metric and report the per-model results in the revised version, with calculation details and the all-six-facts check in the appendix.

**W2: Prompt imbalance.** We understand the reviewer's concern. We included this sentence to keep the neutral summary focused on the supplied evidence and discourage outside additions. To check whether it drove the results, we removed only that sentence and generated new neutral responses for Qwen3-8B and Llama-3.1-8B on all 160 scenarios. We reused the original goal responses and recalculated distortion scores.

*Selection* changed little in absolute terms, and *framing* was almost unchanged. *Ordering* decreased but remained statistically significant:

| Measure | Qwen: original → sentence removed | Llama: original → sentence removed |
|---|---:|---:|
| Selection | 0.0146 → 0.0229 | 0.03125 → 0.03125 |
| Framing | 0.1848 → 0.1799 | 0.2342 → 0.2373 |
| Ordering, using the shared-pair calculation in both columns | 0.0922 → 0.0707 | 0.0851 → 0.0545 |
| Specificity |  −0.0046 → −0.0055 | 0.0269 → 0.0243 |
| Emphasis | 0.0788 → 0.0736 | 0.1457 → 0.1499

After Benjamini–Hochberg adjustment across all ten tests (five dimensions × two models), selection, emphasis, ordering, and framing remain significant for both models; specificity is not significant for either. Thus, the main pattern persists without the grounding instruction. We will include this experiment in the revision.

**C1: *italics* vs Capitalization:** We appreciate this suggestion and will use consistent lowercase and italicized dimension names in the text to aid readability.


## Response to K6Zq

We would like to thank reviewer K6Zq for their careful review and for recognizing the importance of our work and the quality of writing. 

**W1: Interpreting *selection*.** We agree it would also be helpful to report how many favorable and adverse facts are actually included rather than the imbalance on its own. *Selection* measures a change in the balance between the two, so a small score alone cannot establish that the evidence is retained. We checked fact inclusion across all 3,840 responses using the existing sentence-fact matches:

| Supplied facts included | Neutral | Goal-conditioned |
|---|---:|---:|
| Favorable facts | 99.31% | 98.85% |
| Adverse facts | 99.60% | 98.52% |
| Responses including all six facts | 96.77% (1,858/1,920) | 92.86% (1,783/1,920) |

Both types of fact are retained at high rates, although inclusion falls under goal conditioning, more for adverse facts (1.08 percentage points) than favorable facts (0.45 points). The aggregate also hides model differences: goal-conditioned inclusion ranges from 98.12–99.79% for favorable facts and 95.83–99.58% for adverse facts. Llama-3.1-8B has the largest adverse-fact decline, from 99.17% to 95.83%. These results support high overall retention, not complete retention or an absence of selective omission. We will revise the claims based on *selection* alone and report the full per-model results in the appendix of the revised version.


**W2: Ordering and missing facts.** We agree that the paper’s formula for ordering was affected by fact omissions, and have since updated it. For the sake of brevity we direct the reviewer to the **W1 response to reviewer Vvyc**. We belive this also addresses the reviewer's request for a robustness check as that is now integrated in the *ordering* formula. We have also conducted further analysis on scenarios where all six facts appear in both responses and found positive mean effects for every model (also in W1 response to reviewer Vvyc.)



## Response to GTaK

We thank the reviewer for recognizing the importance of our research question and the experimental rigor of our work. Below we address each of the two weaknesses in order of appearance in the review. 

**W1: Grounding, roles, and audiences.** We agree that the prompts differ in more than the objective sentence and that is partially intentional. The grounding instruction was intended to keep the neutral response focused on the supplied evidence. We tested its effect by removing only that sentence and generating new neutral responses for Qwen3-8B and Llama-3.1-8B across all 160 scenarios, while reusing the original goal responses.

*Selection* changed from 0.0146 to 0.0229 for Qwen and stayed at 0.03125 for Llama. *Framing* was almost unchanged: 0.1848 → 0.1799 and 0.2342 → 0.2373, respectively. Using the shared-pair *ordering* definition for both comparisons, *ordering* decreased from 0.0922 to 0.0707 and from 0.0851 to 0.0545, but remained statistically significant. Both *ordering* and *framing* survive adjustment for the ten tests. The main pattern therefore persists in these two models without the asymmetric grounding instruction (see response to W2 of reviewer Vvyc for further details).

We deliberately retained the roles and audiences because together they establish why the communicator has a stake in the recipient's decision and why particular facts matter to the recipient. We also note that both the recipient and the scenario are provided in both conditions. For example, an enrollment specialist represents an institution seeking increased enrollment, while particular adverse facts may be especially important for a particular recipient. This is the institutional setting we aim to study. Without the role and recipient context, the scenario would no longer instantiate the institutional tension that JANUS is designed to study. We also do not treat persuasion itself as misleading: a recommendation can present the evidence faithfully. Our human validation also measures persuasion and misleadingness separately: for high-JANUS pairs, annotators more often selected the goal-conditioned output as more likely to induce the favored action and as creating a more favorable impression than the complete evidence warranted (Table 10).

**W2.1: Ordering and omission.** We agree and have recalculated *ordering* using only favorable–adverse pairs present in both responses, comparing their first mentions and assigning half credit to same-sentence ties. Missing pairs are excluded from both sides. All 12 models retain positive mean effects, and 11 are significant under the paper's testing procedure and 60-test adjustment. GPT-5.4 Mini remains positive but uncertain (0.0354; 95% interval [−0.0028, 0.0747]). Only one scenario each for DeepSeek v4 Thinking and Llama-3.1-8B has no shared pair and is excluded. A further check on scenarios retaining all six facts in both responses also gives positive means for every model. We will update the metric and include the full results in the revised version (see response to W1 of reviewer Vvyc for the formula and further details).

**W2.2: Numerical specificity.** We agree that our measure captures the retention of numbers, rather than every kind of qualification. We chose numbers because their preservation can be measured consistently and reproducibly. We will make this scope explicit by calling the dimension “numerical specificity.”

We also agree that losing a number is not necessarily goal-favoring. However, the direction does not follow simply from whether the fact is favorable or adverse. Removing the size of a cost can downplay a risk; replacing “returns increased by 5%” with “returns increased dramatically” can exaggerate a benefit. In both cases, the reader loses information needed to judge the magnitude of the claim. This is why we measure numerical loss in the same way for both types of fact. The measure asks whether quantitative detail is preserved; *framing* separately assesses whether the wording favors the goal.

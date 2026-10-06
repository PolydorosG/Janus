Reviewer Vvyc

Paper Summary:
The authors present JANUS, a benchmark designed to measure how an institutional objective changes the way LLMs communicate a fixed pool of stipulated-true evidence. The benchmark includes 160 synthetic decision scenarios across eight domains, with each scenario containing three favorable and three adverse facts, and evaluates 12 model configurations under paired neutral and goal-conditioned prompts. The evaluation considers five dimensions of communicative distortion: selection, emphasis, ordering, specificity, and framing. The reported results show relatively small changes in Selection, while Ordering and Framing exhibit larger and more consistent shifts, with Emphasis varying more across models.

Summary Of Strengths:
The paper addresses an under-explored failure mode, showing that factually grounded outputs can still shape a recipient’s impression through selective presentation rather than explicit falsehoods.

The decomposition into selection, emphasis, ordering, specificity, and framing provides a fine-grained view of how communication changes under goal conditioning.

The empirical evaluation is reasonably broad, covering 160 scenarios across eight domains and 12 model configurations from multiple model families

The paper devotes substantial effort to benchmark quality control and evaluation validation, including manual review, agreement analysis, and robustness checks for the automated judges.

Summary Of Weaknesses:
The Ordering metric may partially conflate omission with reordering, since favorable–adverse pairs in which one of the two facts is absent contribute zero to the adverse-first rate while still remaining in the denominator, which can increase the oriented score even when no actual ordering relation is observed.

The paired prompting setup is not fully controlled, since the neutral prompt explicitly requires the model to use only the provided facts and avoid outside information, whereas the goal-conditioned prompt does not retain the same grounding constraint, making it harder to attribute the observed differences solely to the institutional objective.

Comments Suggestions And Typos:
The paper is generally well written and easy to follow, with a clear presentation of the benchmark design, evaluation dimensions, and main findings. As a minor stylistic suggestion, the authors could consider using italics rather than capitalization when referring to the individual evaluation dimensions, which may make these terms stand out more naturally in the text without interrupting the reading flow.


Reviewer K6Zq

Paper Summary:
The paper introduces JANUS, a benchmark for goal-conditioned information distortion in fact-grounded LLM communication. JANUS allows analysis across 5 dimensions (namely, selection, emphasis, ordering, specificity, framing), contains 160 scenarios (each containing a pool of 3 favorable and 3 adverse facts) across 8 domains, and is evaluated on 12 model configurations. The main empirical result is that goal conditioning has relatively small effects on the Selection metric, while larger and more consistent changes appear in Ordering and Framing, with Emphasis showing more model-dependent behavior.

Summary Of Strengths:
The paper targets an important and relatively underexplored failure mode: factually grounded communication that can still create a misleading impression through selective presentation. The paired evaluation setup is a useful design choice, since the same scenario, evidence pool, and evidence order are used across neutral and goal-conditioned generations. The evaluation has reasonably broad coverage, spanning 160 scenarios, eight domains, and exploring 12 model configurations across different sizes and model families. The work includes human validation showing that higher JANUS scores are associated with greater human-perceived misleadingness.

Summary Of Weaknesses:
Near-zero Selection does not necessarily imply that models retain the material evidence, since the metric measures favorable-vs.-adverse coverage imbalance rather than absolute coverage. Reporting absolute coverage for both classes would make this conclusion more convincing.

The Ordering metric appears sensitive to fact omission, since pairs for which either fact is absent contribute zero to the adverse-first rate while remaining in the denominator. A robustness analysis conditioning on pairs where both facts are present would strengthen the main Ordering result.

Comments Suggestions And Typos:
The paper is well written and easy to follow. Tables and figures are readable. Reported examples and prompt (both in the main body and appendix are useful).


Reviewer GTaK

Paper Summary:
JANUS introduces a benchmark for measuring how an objective changes an LLM's presentation of a fixed pool of stipulated-true evidence. It contains 160 synthetic decision scenarios across eight domains, each with three favorable and three adverse facts, and evaluates 12 model configurations under paired neutral and goal-conditioned prompts. The evaluation decomposes the change into selection, emphasis, ordering, specificity, and framing, using automated sentence-fact matching and framing judgments with manual audits. The effects seem strongest for ordering and framing, and a small prompt-level guardrail is explored as an initial mitigation measure.

Summary Of Strengths:
JANUS targets an important and under-evaluated problem: a response can be literally accurate while still steering a recipient through omission, ordering, emphasis, or framing. The fixed six-fact setup and paired prompts are conceptually useful for separating some pragmatic effects from ordinary hallucination, and the manuscript gives definitions and prompts for five dimensions. The multi-domain/model evaluation, dataset quality-control procedure, matcher audit, cross-family framing check, qualitative examples, and mitigation experiment are valuable steps toward a reusable benchmark. The paper is generally readable and relevant to NLP, model auditing, and pragmatic communication.

Summary Of Weaknesses:
The neutral system prompt says "Use only the provided facts and do not add outside information," whereas the goal-conditioned prompt removes this instruction and additionally changes the role and audience. The observed differences may therefore reflect a changed grounding constraint, role/audience conditioning, or explicit persuasion, rather than the objective alone.

The ordering score divides by all nine favorable-adverse pairs, while pairs with an absent fact contribute zero; omission can consequently appear as favorable ordering even when the co-mentioned facts retain the same order. Specificity is only unsigned numerical retention, ignores non-numerical qualifications, and is not inherently goal-favoring: dropping a favorable number and dropping an adverse number have opposite implications.

Comments Suggestions And Typos:
The manuscript is clearly written, and the qualitative paired examples make the proposed phenomenon intuitive. "Specificity" should be renamed or explicitly qualified as numerical retention unless a broader measure is added. No major typographical problems were observed in the supplied transcription.



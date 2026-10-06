Official Review of Submission10170 by Reviewer qNBU


Paper Summary:
ThispaperintroducesJANUS,abenchmarkforevaluatinggoal-conditionedinformation distortioninLLMcommunication.Unliketraditionalhallucination,factuality,orexplicit deceptionbenchmarks,JANUSfocusesonwhethermodelscanproducetruthfulyet misleadingcommunicationbyselectivelypresenting,emphasizing,ordering,specifying,or framingthesamefixedpooloftruematerialfacts.Thebenchmarkcomparesmodeloutputs underaneutralconditionandagoal-conditionedcondition,whereaninstitutionalobjective maycreatepressuretofavoradoption,approval,enrollment,support,oracceptance. Experimentsacross160scenarios,8domains,and12LLMsshowthatmodelsusuallydonot directlyomitadversefacts,buttheyoftenshifttherecipient’snetimpressionthrough ordering,framing,andsometimesemphasis.Overall,thepaperarguesthatfactualcorrectness aloneisinsufficientforevaluatingfaithfulLLMcommunication.
Summary Of Strengths:
Thepaperaddressesanimportantandunder-exploredproblem:LLMsmayproducetruthful yetmisleadingcommunicationwithoutgeneratinghallucinationsorexplicitfalsehoods.
Thepairedbenchmarkdesigniswellmotivated.Byholdingtherecipientprofileandfixed factpoolconstantacrosstheneutralandgoal-conditionedconditions,JANUScanbetter isolatedistortioncausedbyinstitutionalobjectives.
Thefiveproposeddistortiondimensions―selection,emphasis,ordering,specificity,and framing―areintuitiveandinterpretable,coveringseveralkeywaysinwhichcommunication canbecomemisleadingwhileremainingfactuallygrounded.
Theempiricalfindingsaremeaningful.Thepapershowsthatmodelsusuallydonotsimply omitadversefacts,butinsteadshiftthereader’snetimpressionmainlythroughordering andframing.
Thepaperincludesusefulvalidationofitsautomatedevaluationpipeline,especiallythe sentence‒factmatchingandframingjudge,whichimprovesthecredibilityofthereported results
Summary Of Weaknesses:
The construction of the fact pool may involve substantial subjectivity. Thedistinction betweenfavorableandadversefactsdependsontheauthors’judgment,andinmany scenariosafactmayhaveambiguousorcontext-dependentvalence.Althoughthepaper reportsannotationcalibrationandagreement,itdoesnotsufficientlydiscusshowsuch ambiguouscasesarehandled.
TheevaluationofframingheavilyreliesonanLLM-as-a-judgepipeline.Sinceframingisone ofthepaper’smostcentralfindingsbutalsothemostsubjectivedimension,errorsor biasesinthejudgecouldaffecttheconclusions.Thisconcernisfurtheramplifiedbythefact thatQwen3-8BisusedasthejudgewhileseveralQwenmodelsarealsoincludedamongthe evaluatedsystems,raisingthepossibilityofmodel-familyorstyle-relatedbias.
Thepaperlacksmitigationorrobustnessexperiments.Whileitshowsthatgoal-conditioned promptscaninducedistortion,itdoesnottestwhethersimpleinterventions,suchasexplicit balanceinstructions,risk-firstordering,numerical-detailpreservation,orstructuredoutput formats,canreducesuchdistortion.
Theinterpretationofstatisticalsignificanceandeffectsizecouldbeclearer.Somereported distortionscoresarerelativelysmall,sothepapershouldbetterexplainwhatthesedeltas meaninactualgeneratedtextandwhethertheycorrespondtomeaningfulchangesin human-perceivedmisleadingness.
Comments Suggestions And Typos:
Thepapercouldexplainthepracticalmeaningofthedistortionscoresmoreclearly, especiallywhenthereportedvaluesaresmall.Itwouldbehelpfultoshowwhetherthese numericalchangescorrespondtomeaningfuldifferencesinactualgeneratedtextorhuman- perceivedmisleadingness.
Theauthorscouldaddsimplemitigationexperiments,suchasexplicitlyrequiringbalanced presentationoffavorableandadversefacts,preservingallnumericaldetails,orusinga structuredtableformat.ThiswouldshowwhetherJANUScanalsobeusedtoevaluate mitigationstrategies.
Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 2.5
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 3 = Findings: I think this paper could be accepted to the Findings of the ACL.
Ethical Concerns:
There are no concerns with this submission
Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 3 = Potentially useful: Someone might find the new datasets useful for their work.
Software: 1 = No usable software released.



Official Review of Submission10170 by Reviewer foxd
Official Reviewby Reviewer foxd02 Jul 2026, 12:22 (modified: 09 Jul 2026, 01:59)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer foxd, AuthorsRevisions
Paper Summary:
The paper presents a benchmark and a framework for evaluating whether LLMs distort information when instructed in a goal-oriented manner. They evaluate 160 scenarios across 8 domains, on a range of 12 LLMs varying in architecture and model sizes. They report that LLMs are sensitive to framing objectives.
Summary Of Strengths:
The core motivation of the paper to evaluate whether LLMs hide distortions within factual contexts to support a specific goal is quite valuable.
The behavioral scoring dimensions are intuitive and explained with clear formulations.
I found the paired prompting strategy very useful as it allows to isolate the findings from model specific patterns.
The evaluation is done across model archetechtures and model sizes.
Summary Of Weaknesses:
Dataset construction:
For generating the factual statements, authors manually define domains, recipient profiles, decision subjects, institutional actors, and objectives. It is unclear from the text what criteria was used by authors to select the items in a specific dimension. Given that the list of values these dimension are not provided in the paper, it is hard to make an intuitive guess regarding this decision process.
The prompt used for generating the facts are not provided. It would be interesting to know how the model was prompted to generate the favorable and adverse facts. In this context I am also concerned about how a "fact" is defined in the prompt and in general through out the paper. Since these are not true statements about real world, it would have been better if authors acknowledged that these are synthetic facts.
During the manual verification stage, if a fact statement was found to be not usable, was the complete scenario discarded? It would be interesting to know the quality of the 20 scenarios that were manually evaluated. The distribution of the initial vs. final set of facts across domains would also be valuable.
Authors measure "superficial artifacts" in the two sets of facts, but they do not report exactly how this was measured. Specifically numerical-detail frequency and hedging frequency.
Framing objective: The observed effect on framing seems to be quite obvious. Since the model is prompted to frame it in a goal-oriented way, the reported deviation from the neutral prompt seems to be expected. Also, the framing judge seems to take into account a given fact, but the judgement as I understand is only made regarding the framing. So it is not clear if the facts were actually distorted or not.
Related work: I understand that due to space issues authors have moved the related work to appendix. However, I am of the very strong opinion that to contextualize the current research a comprehensive related work section is indispensable in the main content. Also, I find the related work section in the Appendix does not sufficiently cover the related literature. For instance, the evaluation benchmarks listed in Table 1 is not linked to its corresponding research papers (also not in the appendix).
The main content of the paper should be self contained. To review the paper fairly, it was necessary to read the appendix as well.
Overall the paper needs to include details regarding the created dataset with concrete examples and clarification on how the quality of the instances were ensured.
Comments Suggestions And Typos:
Presentation:
Font size of text in Figures are very small compared to the paper font making it very hard to read especially in A4.
Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 2.5 = Borderline Findings
Ethical Concerns:
There are no concerns with this submission
Reproducibility: 1 = They would not be able to reproduce the results here no matter how hard they tried.
Datasets: 3 = Potentially useful: Someone might find the new datasets useful for their work.
Software: 1 = No usable software released.



Official Review of Submission10170 by Reviewer Raq4
Official Reviewby Reviewer Raq430 Jun 2026, 06:21 (modified: 09 Jul 2026, 01:59)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer Raq4, AuthorsRevisions
Paper Summary:
This paper aims to explore the selective treatment of true material facts, including omitting adverse evidence, softening unfavorable details, emphasizing favorable details, or replacing precise qualifications with vague language. To this end, the authors propose JANUS to evluate the goal-conditioned pragmatic distortion in fact-grounded LLM outputs.
Summary Of Strengths:
This paper focuses on an interesting research question that can fill the gaps in existing works.
Summary Of Weaknesses:
This paper is hard to follow.
The authors do not give a clear definition of this task. In my understanding, the authors want to design a function  that scores a given output  generated by LLMs. Meanwhile, the  covers five domains. However, I do not understand why the authors design two different prompts (eq 1). Meanwhile, what are the differences between , , and ?
The authors do not provide any examples, which makes it impossible to understand the authors' aims.
In this paper, the authors employ many symbols and equations. However, the authors do not provide sufficient explanations. For example, for eq (1), the authors claimed that  denotes the recipient profile. But what is the recipient profile? It is the first time this term appears without any additional explanation (to my knowledge, it is not a fixed term). As well as  (what is institutional objective).
Thus, I suggest the authors proofread this version and make it at least express the idea clearly.
If my understanding is correct (that the authors aim to design a functional score function such as Factscore).
The authors should prove that the proposed function is reliable. The authors should provide more evidence, such as a human evaluation (not the same as lines 403-412).
Besides Table 1, the authors should provide quantified evidence to prove that the proposed function is novel.
What are "160 scenarios across 8 domains"? In my understanding, there are 160 instances in this benchmark? If so, I think the scale is limited. Meanwhile, the authors do not prove that the 160 instances are of high quality. As a result, the proposed benchmark cannot reliably evaluate the ability of LLMs, in my opinion.
The main text should be self-contained. The authors seem to provide much content in the Appendix. However, there is no reference to the appendix. This makes it impossible for readers to know what is in the Appendix.
In summary, I think this paper is far from being publishable; thus, I have decided to give it a score of 2.
Comments Suggestions And Typos:
Pls see the weaknesses
Confidence: 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 2.5
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle.
Ethical Concerns:
There are no concerns with this submission
Needs Ethics Review: No
Reproducibility: 1 = They would not be able to reproduce the results here no matter how hard they tried.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.

# Annotation guideline: legitimate vs. annoying "rather than"

For each sentence, the highlighted **"rather than"** sits inside a real sentence
pulled from a real paper (2019 ACL Anthology or 2026 arXiv NLP papers). Judge
only this specific use of the construction.

## Labels

- **legitimate** — "rather than X" does real work: it names a genuine
  alternative the reader would otherwise assume, clarifies a design/methodological
  choice, or draws a contrast that matters for understanding the claim. Test: if
  you deleted "rather than X", would the sentence lose information the reader
  needs? If yes → legitimate.

- **annoying** — this use of "rather than" annoys you. Go with your reaction;
  there is no need to work out why.

- **unsure** — genuinely ambiguous, or the sentence assumes context you don't
  have (e.g. it clearly refers to something earlier) and you can't tell either way.

- **garbled** — the extracted text itself is broken (PDF column-merge,
  table/figure text mixed into the sentence, mid-word line breaks) badly enough
  that you can't actually read the sentence as written English. Not a judgment
  about the antithesis — a judgment about extraction quality.

## What NOT to judge

Not the paper's quality, not whether you agree with the claim, not grammar
elsewhere in the sentence. Only: does *this* "rather than" earn its place.

## Calibration examples

**legitimate** (real, from the sample):
> "When a paper reports multiple GPU configurations, we use the largest
> configuration rather than summing across all configurations."

A real methodological choice between two concrete alternatives; the reader
needs to know which one was taken.

> "Several widely used video benchmarks follow the same distribution model,
> providing pre-extracted features, annotations, and evaluation tooling rather
> than raw video files."

Names a specific, non-obvious alternative (raw video files) that a reader
might otherwise assume was provided.

**annoying** (constructed illustration — the real sample may have few or many
like this; that's part of what we're measuring):
> "Our approach focuses on producing accurate, reliable results rather than
> inaccurate, unreliable ones."

## Format

One label per item, independently — don't look at other items' labels or try
to guess the source paper/era.
## Your task

Label every item below using the guideline above. Output your answer as
plain text, **exactly one line per item, in this exact format and this
exact order**, nothing else before, between, or after:

item_001: legitimate
item_002: annoying
item_003: unsure
...

Use only the four labels: legitimate, annoying, unsure, garbled. Do not
add explanations, headers, blank lines, or renumber -- one line per item
below, in the order given, or the response can't be scored.

## Items
item_1865: The release includes all 1,145 of them,
so the question is open over material already distributed rather than a decision still pending.
item_1867: Inter-annotator agreement was Fleiss' , which is substantial, so the ranking reflects genuine quality differences rather than scorer leniency.
item_1868: We rerun the matcher rather than
post-hoc relabeling the matching-evaluation output, because the original prompt
does not ask the model to expose this distinction.
item_1869: The consistency of the drop across four metrics, rather than a large drop in one, suggests AEA's effect is distributed evenly, correcting many small cross-modal disagreements rather than a few large ones, consistent with the mechanism's design (Section [redacted]) as a per-claim argmax over modality-weighted confidence rather than a global override.
item_1870: The main analysis assigns each sentence its top-1 label, so a gap could in
principle reflect where the decision boundary falls rather than the underlying
distribution.
item_1871: The negative result here is therefore a statement about routing at 2-36 success rather than about routing, and it predicts its own reversal: run this measurement on an agent that solves most of these tasks and the label supply and the contested set grow together.
item_1874: Accuracy is less central than CE
and JSD because the weak target is a soft distribution rather than a hard gold
label.
item_1877: The moderate overlap between [redacted] and [redacted] confirms 
that [redacted] extends rather than duplicates the perceived quality 
signal.
item_1878: Third, the rank-biserial effect for peak intensity on Llama-8B () is below the conventional small-effect threshold (), indicating that intensity is not a portable quality signal across architectures and should be reported alongside count and ratio rather than in isolation.
item_1884: The following limitations define the scope of our claims rather than qualify the contributions above.
item_1890: Successive post-training stages suppress the focal conflict and surprise-curiosity families and inflate neutral content, overshooting human prevalence rather than converging on it.
item_1892: With [redacted] architectures, this is descriptive rather than an established tradeoff, but it has methodological implications: cross-architecture comparisons of stability should be reported on the with-peaks subset rather than on the full 200 problems, since imputing [redacted] for no-peak problems would conflate the no-peak phenomenon with the within-with-peaks-stability phenomenon.
item_1895: Rules with fewer than 8 pooled conditional hits are omitted, and the largest omitted enrichment is stated per block rather than left to the reader.
item_1897: First, it localizes the tool schema shown to the agent: descriptions, enum choices, and examples are rendered in L2 so the scenario tests target-language tool use rather than English schema reading, while the underlying implementation remains unchanged.
item_1902: The results confirm that LLM-XTM is model-agnostic rather than tied to a single proprietary engine.
item_1903: LLaVA exhibits a milder variant in which female signals compress toward zero in late layers rather than crossing into male territory.
item_1904: One open question is whether BAFA performs better even for such substantially larger black-box models, since our C-ERM step uses a comparatively small BERT surrogate to reduce the fairness-metric version space induced by queried scores rather than the black box’s full parameter space; we address this question empirically in this case study.
item_1906: Second, it should be constructive: the resulting method [redacted] must represent a significant semantic advancement from the problem context [redacted] towards the solution space, rather than a mere restatement of the input.
item_1907: Rationale for the Simulated Clarification Interface

While we acknowledge that simulated interactions cannot fully capture the stochastic nature of human behavior, our design choice to utilize a Dynamic Clarification Interface rather than human-in-the-loop evaluation is deliberately driven by the specific requirements of constructing a robust benchmark.
item_1912: MUSE therefore adopts a five-level taxonomy that emphasizes capability transfer rather than surface tone: Compliance (harmful capability directly transferred), Partial Compliance (incomplete but still actionable harmful information), Indirect Refusal (avoids assisting without explicit refusal), Direct Refusal (explicitly declines), and Non-Responsive (irrelevant output).
item_1917: To ensure coverage, rules were deliberately designed to err on the side of over-filtering (i.e., allowing more false positives rather than false negatives).
item_1919: Experiment 1: All metrics except for Sent-NLL are biased in different ways

To ensure that observed metric differences reflect the metrics themselves rather than model-specific confounds,(For example, a multilingual model trained predominantly on Chinese may assign lower BPC to Chinese text, masking BPC's inherent bias against logographic writing systems: Chinese characters encode more information per character than alphabetic scripts, artificially shortening sentences and improving BPC scores.) we use monolingual models trained on parallel corpora with varying vocabulary sizes.
item_1921: Interestingly, no model performs uniformly well across all setups, suggesting operation-specific strengths and weaknesses rather than a single transferable notion of phrase-level competence.
item_1922: We therefore report category-level scores in the experiments rather than relying only on one aggregate score.

[!t]
category distribution by domain.

*[!t]
*

0.35em

*
Main leaderboard.
item_1928: Contributions.
(i) We introduce , the first fanfiction-register jailbreak
family, using twelve real AO3 subgenres as universal attack
carriers, and find that it roughly triples the attack success
rate of six existing baselines while remaining positive on every
model and surviving length matching.
(ii) Through a style transfer experiment we show that
the choice of conditioning corpus dominates the choice of
structural overlay: the template-free plain cell alone matches
the best existing overlay.
(iii) We find that two defenses widen the vernacular-to-baseline ratio rather than narrowing it, which means template-targeting defenses simply steer attackers toward register attacks.
(iv) We propose , a static four-turn attack pipeline that uses neither an adversarial attacker LLM nor optimization, yet attains a mean ASR of 0.924 and exceeds three existing multi-turn methods.
item_1929: We emphasize that our approach is intended to support, rather than replace, educators.
item_1932: This consistency suggests that the rubric-based evaluation captures stable differences in model performance rather than judge-specific preferences.
item_1938: To enhance my understanding, I kindly request your insights regarding the decision to opt for 14 layers instead and the possible reasons behind the relatively higher bpc despite employing deeper layers.",
        "answer": "The observed difference between the reported bits per character (bpc) for Enwik8 in Section 4.3 of our paper and the original Transformer-XL paper can be attributed to our decision to utilize Nvidia's implemented Transformer-XL (https://catalog.ngc.nvidia.com/orgs/nvidia/resources/transformerxl_for_pytorch) rather than the official repository.
item_1941: The hierarchical nature of human thinking: humans usually explicit reasons and plans at multiple levels of abstraction [1][2], i.e., first decide a strategy, then formulate the detailed response:
[1] A hierarchy of intrinsic timescales across primate cortex, Nature Neuroscience 2014 [2] Intrinsic timescales in the visual cortex change with selective attention and reflect spatial connectivity, Nature Communications 2023

Selection of Strategies: Rather than being dataset-specific, the strategies employed in the paper are grounded by well-established theories, which ensures their applications in the related fields:
Dailydialog: the Speech Act Theory proposed by John Searle (1975)
ESConv: the Helping Skills Theory proposed by Clara Hill (2009)
Framework Extendability: We provide an HRL-based framework that can make hierarchical levels of planning for conversation tasks, given a set of theoretically grounded strategies.
item_1945: The paper can cite this study as a companion artifact rather than including it in the body.
item_1946: Conversely, AlphaEdit [redacted] enforces stability through geometric constraints rather than architectural changes.
item_1947: Both hierarchical methods outperform the flat baselines, but [redacted] extends RAPTOR's gains by a further 9 on Factuality and 15 on CtxR: cluster summaries surface thematic connections flat embeddings miss [redacted], letting the agent match queries to relevant document groups rather than passages.
item_2094: The comparison therefore supports a granularity-context-efficiency trade-off rather than universal superiority of one representation.
item_2084: This pattern suggests that structured state tracking helps maintain an explicit reasoning status for retrieval and refinement, while fine-grained template editing updates reusable memory locally rather than rewriting the whole reasoning scaffold.
item_1781: Under greedy decoding, [redacted] is a deterministic function of [redacted] for a fixed annotator, so reported variability is across-article rather than within-article.
item_1836: This is important because, unlike the label-based counterfactual, the payoff structure here actually changes; consistently reaching 96 therefore suggests genuine sensitivity to altered incentives rather than mere invariance to relabeling.
item_1925: While these approaches effectively mitigate factual hallucinations, they tend to treat errors primarily as knowledge deficiencies rather than intent deficiencies. [redacted] As a result, they remain insufficient when the primary source of failure is misunderstanding the user's intent.
item_2070: This is consistent with our case study, in which LLM-extracted
entities more precisely identify the central topic of a passage, particularly
when the title refers to a sub-entity rather than the primary subject.
item_1848: The consistency across guidance models 
confirms that effectiveness stems from the representation-space 
alignment mechanism rather than properties specific to any single guidance 
model.
item_2135: Rubric-Transfer Fidelity

We further examine whether the Phase-I rubric-transfer process faithfully operationalizes the source rubric rather than introducing unsupported scoring criteria.
item_2142: Representative examples include systems that manage long-term context or learn read/write policies (e.g., MemGPT, [redacted]), and methods that store reflective summaries or compact experience units rather than full trajectories, such as Reflexion [redacted], A-MEM [redacted], and Memento [redacted].
item_1809: Carrying the literal numerator (≥5/8 = 62.5) instead would let DOM+stext clear two metrics and this negative would appear to break, so the choice is load-bearing and is disclosed rather than defended.
item_1826: It therefore describes sampling uncertainty on that split, not run-to-run variation: the two are distinct, and we keep them separate rather than pooling seeds, since our three seeds re-decide the same items and their decisions are not independent draws.
item_1887: Here [redacted] may name an assumption rather than a recorded fact: information
to be obtained rather than a change in the patient.
item_2113: DualMem's cluster CIs are tight at every [redacted] on both back-ends, reflecting that its lead is consistent across chain trunks rather than carried by a handful of outliers.
item_2106: As LLM-driven agents, [redacted] inherits LLM limits on questions lacking a clear scope (App. [redacted]), assisting scholars rather than replacing them.
item_1789: Therefore, users should interpret the outputs of STEM as a reflection of the facts stored in the specific Knowledge Graph, rather than as an unbiased representation of real-world truth.
item_2010: The following
desiderata characterize metrics that gauge the fidelity of P with respect to R rather than just goal
completion.
item_1283: Thus, for illustrative purposes, we are using more standard quantifiers i.e. dare
‘who’ and nani ‘what’ types of quantifiers, rather than dono-quantifiers.
item_1934: It is intended to complement the per-system results in the main text by showing the domain-level distribution of ASR rather than individual model or agent-scaffold behavior.

*[h]
             Average ASR by risk domain.

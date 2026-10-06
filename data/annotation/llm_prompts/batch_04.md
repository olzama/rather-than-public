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
item_1802: The presentation follows processor roles rather than engineering stage names.
item_2026: In terms of prompting stability, Claude 3.5/3.7 and Llama 3.3 are generally the most consistent across ZS, CoT, and SPP, whereas Mistral Large exhibits the highest variability and Claude 4 as well as DeepSeek R1 are more sensitive to prompting, particularly in settings where additional reasoning appears to induce overthinking rather than better strategic adaptation.

*[h!]

Cross-model qualitative comparison across games, counterfactual robustness, and efficiency.
item_1977: We hypothesise that this is because of the lexicon
which includes primarily MSA terms and Egyp-

Classifier
Ridge Classifier
Logistic Regression
Passive Aggressive
Linear SVC
SGD Classifier
Multinomial NB
Bernoulli NB
Complement NB

2 classes
73
74
73
73
73
74
72
75

Table 12: Accuracy of the proposed model on binary
classification trained and tested on Shami-Senti

tian terms rather than Levantine sentiment terms
so the probabilities of features are less accurate.
item_2040: The single cell that changes, classifieds · B1, loses a class rather than gaining one: the prior-order label keeps DOM and SoM above the threshold, the measured-cost label concentrates enough of those rows onto Vision that only Vision survives.
item_2122: Rather than treating this as a fixed taxonomy from the start, the definition evolved through calibration discussions, and experts converged on nine binary criteria (-), phrased as yes/no questions, to capture recurring procurement rationales and make disagreements observable rather than hidden under a single label.
item_1756: The computational cost of the pipeline scales with the number of hypotheses and validation tests, making the approach best suited for post-hoc audits rather than real-time monitoring. [redacted] We view the pipeline as complementary to existing benchmark-based evaluations and targeted testing frameworks, rather than as a replacement for them.
item_1801: We optimize the standard conditional language-modeling objective:

Thus, SFT stores target-annotator behavior in independent adapted parameters rather than recovering it from a profile string or symbolic annotator ID.
item_1818: It does not spread evenly.
[redacted] report flip rates that vary by task category, on tests they call
underpowered (too few questions per category), so an aggregate figure is misleading rather than
imprecise.
item_1970: For instance, the expression may be
interpreted as an insult rather than as, for instance,
humorous break to a heated discussion.
item_2101: Grasp's probe compute is not budget-matched to the baselines, so its advantage over them could in principle reflect the extra compute the gate consumes rather than the gate's decisions themselves.
item_2102: Rather than treating inference as a static process, the Student iteratively updates its policy by learning from both the original test question and the variant questions synthesized by the Teacher.
item_1900: This requires relative judgment capability, the ability to compare and rank audio samples by their acoustic similarity to a reference speaker, rather than making absolute binary decisions.
item_2030: ARC ECE increases slightly in the forward ordering across all three smaller models (e.g., Llama: 0.095 [redacted] 0.112), but this effect reverses under alternative orderings (Appendix [redacted]), indicating a domain-sequencing artifact from cumulative LoRA weight transfer rather than a systematic limitation.
item_1839: Options A-D are contentful answers; E denotes "information not available," so a model must recognize missing memory rather than guess among four plausible kitchen actions-otherwise parametric priors inflate accuracy on underspecified probes.
item_1939: We therefore study latent multi-hop reasoning by tracing whether bridge entities [redacted] emerge in hidden states, and where they appear [redacted].





*[!ht]
            An illustration of latent multi-hop reasoning in LLMs from a probabilistic perspective. 
(a) Electron behavior follows a probabilistic distribution rather than fixed orbits. 
(b) Hop-aligned circuit hypothesis assumes layer-by-layer recall of bridge entities. 
(c) Observed layer-order inversion, where later-hop entities may emerge earlier than bridge entities. 
(d) Probabilistic recall-and-extract framework combining vertical and horizontal recall, with last token shallow recall serving as an intuition-like signal for answer selection.
item_1955: In the
first one, Arabic-PUD (ar pud), lemmas are romanized, i.e. presented in Latin rather than Arabic
script.
item_1220: MultiLing 2019: Financial Narrative Summarisation
Mahmoud El-Haj
School of Computing and Communications
Lancaster University
United Kingdom
[email]

Abstract

This can happen by detecting narrative sections
that usually includes the management disclosures
rather than the financial statements of the annual
reports.
item_1235: For concrete examples of
DII and SIS from VIST, we refer readers to Figure
1, where Sentence Sets 1 and 2 (see Section 1) are
from the DII and SIS subsets, respectively.

proper sequences for existing story sentences,
rather than on generating those sentences themselves.
item_2034: The same model run produces both this transliteration output and the English translation in Table [redacted]; the two share the subjectivity (P, R, F1) classifier output. chrF++ is the appropriate character-level metric for a Romanization target; COMET is not applicable here, as its estimator is trained for translation into a natural language rather than for a fixed transliteration convention.
item_2146: After the release of the leaderboards, we also test a new pair-generation variant that uses two prompt settings per input rather than a single prompt: one that encourages a prediction of 1 and another that encourages a prediction of 0.
item_2118: Thus, question-answering and dialogue history are complementary rather than isolated views of each student.
item_1894: A phase is a narrative span in which the character occupies a relatively stable position along the axis, with the number of phases determined by the narrative evidence rather than fixed in advance.
item_1927: Because these approaches optimize different aspects of segmentation quality, we recommend evaluating and selecting segmenters using criteria that match the intended analysis, rather than assuming a single best method.
item_1986: In AL, rather than training on a set of labeled
data sampled at i.i.d. random from some larger
population, the learner engages the annotator in
a cycle of learning, iteratively selecting training
data for annotation and updating its model.
item_1793: Rather than exposing actionable content, we summarize each response by its high-level reasoning structure, objective adherence, and failure patterns.
item_2105: This finding implies that architectural improvements targeting cultural understanding should focus specifically on higher-layer reasoning rather than general visual capability.
item_2078: Rather than relying on a single end-to-end extraction step, the conversion process is decomposed into four conceptual stages: (1) schema-constrained clinical entity extraction, (2) temporal episode construction and normalization, (3) temporal reconciliation and node canonicalization, and (4) typed relation construction and graph finalization.
item_1958: In an effort to derive a closed form for p?X
(rather than solve a Riemannian optimization
problem), we conjecture that the following expression is a good approximation.
item_1875: Second, it is told to judge by mechanism and attack strategy rather than by surface overlap, with explicit guidance that lexical overlap on concrete entity tokens such as names, addresses, or dates is insufficient to admit a cell.
item_1827: Across these dimensions, however, the character is treated as a static target-an identity to be reproduced rather than one whose behavior shifts as events accumulate [redacted].
item_1974: Introduction

Burmese (Myanmar) script is an abugida system,
wherein basic characters can be modified using diacritics at all directions or can be combined vertically, rather than a simple left-to-right horizontal
writing (Ding et al., 2016).
item_1837: The main paper handles this by limiting peak intensity claims to within-architecture discrimination (RQ4) rather than cross-architecture magnitude comparisons.

[!htbp]
lccccc
Domain & 10 & 25 & 50 & 100 & 200 

Math [redacted] & 1.51 & 0.75 & 0.64 & 0.45 & 0.27 

Code [redacted] & 2.00 & 1.14 & 0.61 & 0.19 & 0.10 

Logic [redacted] & 2.80 & 0.00 & 0.00 & 0.81 & 0.44 

Commonsense [redacted] & 2.48 & 1.73 & 0.72 & 0.58 & 0.35 

Mean peak intensity per problem across HSIC kernel bandwidth , computed only on problems with at least one peak (Qwen-7B, [redacted] per cell).
item_2007: We define an entity mention m as the
surface form of an entity, and a relation mention

2
Throughout the paper, we use “DBpedia” to refer to
its infobox extraction component rather than the DBpedia
knowledge base unless specified.
item_1823: Prior 
off-policy injection methods [redacted] 
target verifiable-reward tasks and inject partial expert 
prefixes rather than complete reference outputs.(Refer to [redacted] for a detailed comparison.) Our setting is also complementary to [redacted], which likewise uses human-written books and trains models to reason about long-form narrative continuation.
item_2138: The consistent upward trend confirms that the policy progressively learns to produce higher-quality intermediate reasoning steps, rather than merely arriving at correct final answers by chance.
item_1751: This design directly operationalizes functional ToM as defined: evaluating the active mapping from explicit mental-state representations to behavioral predictions, rather than surface-level pattern matching against conversational cues.
item_1961: The garden path should be eliminated in sentences such as (5-b), the UNREDUCED condition,
where the words “who was” clarify that the verb
“brought” is part of an RC, rather than the main
verb of the sentence.
item_1920: We exclude a candidate elbow-fixation case from this display because the available device audit labels it low risk rather than confirmed shortcut; the two cases below have direct audit support for shortcut risk.
item_1847: The Qwen-7B-to-Qwen-14B trend (36 to 53) is also Bonferroni-significant in this test family, but the main-paper interpretation treats this as a within-Qwen pattern rather than a universal scaling signature, given that the rate reverses dramatically at Llama-8B rather than continuing to rise.

[!t]
lccc
Test & Statistic & [redacted] & Bonf.
item_2045: Probing deeper discriminative stability, the Select Incorrect condition inverts the task, requiring models to identify all distractors rather than the single valid answer, while None Provided replaces the correct option with the string "None of the provided options is correct," testing the capacity to recognize valid answer absence.
item_2060: The DAG-derived diagnostics consistently improve over them because they evaluate the organization of evidence rather than the amount of search.


@p0.23p0.22Y@
Metric & Meaning & Computation 

logsearchqueries & number of search queries & count query strings in search tool calls, then log1p 

logvisitedpages & number of visited pages & count URLs in visit tool calls, then log1p 

logtracelength & trajectory length & total non-system message characters, then log1p 

searchredundancy & repeated search-query ratio &  

Raw-log baseline features.
item_1972: Also, in 4 of 5 languages, allowing a trained
N OISY C HANNEL (rather than the identity map)
replacing the left one with “ and the right one with ”.
item_2141: Moreover, [redacted] if and only if every adjacent pair [redacted] is co-directional.


*[ht!]

Concept-level torsion is selective rather than uniform.
item_1975: Chinese, as an analytic language, encodes grammatical information in a highly configurational rather than morphological way.
item_1913: Because agreement was enforced procedurally through mandatory joint resolution of every flagged item rather than parallel independent labeling by all four annotators, we do not report a chance-corrected agreement statistic (for example, Fleiss' ); the protocol was designed to eliminate residual disagreement prior to release rather than to measure it post hoc.
item_2039: Rather than claiming priority for "causal" evidence, our study complements this growing line of work by combining four features in a single design: (1) directly dissociating preemption from entrenchment via the +Competing/Competing scheme [redacted]; (2) non-circular partial correlations against human data; (3) a reverse-direction control that diagnoses the asymmetry predicted by preemption theory; and (4) multi-construction scope (dative, causative, locative).
item_2088: For all experiments, we employ Maximal Marginal Marginality (MMR) as the retrieval strategy rather than K-Nearest Neighbors (KNN).
item_2086: Answer-Label Distribution and Balance
Table [redacted] shows that the training labels remain broadly distributed across the valid answer options rather than collapsing onto a single answer position.
item_1969: Rather than making claims based on entire categories of semantic
properties, we base our predictions on underlying
factors involved in the relations between concepts
and properties.
item_2028: Such scenario–answer correlations could provide models with non-evident priors, potentially allowing them to improve performance through prior-based shortcuts rather than image-grounded reasoning.

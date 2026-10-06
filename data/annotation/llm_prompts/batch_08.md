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
item_2042: We further verify that both metrics capture meaningful signals rather than artifacts of structural heuristics or superficial lexical overlap, through controlled analyses (Appendix [redacted], Appendix [redacted]).
item_1951: Another related line of work builds generation
upon sentential exemplars (Guu et al., 2018; Weston et al., 2018; Pandey et al., 2018; Cao et al.,
2018; Peng et al., 2019) in order to improve the
quality of the generation itself, rather than to allow for control over syntactic structures.
item_2076: This design helps ensure that differences across categories are attributable to the underlying non-compliance condition rather than to variations in dataset origin, prompt structure, or taxonomy granularity.
item_2054: Third, the feature analyses remain associational rather than causal.
item_1738: More recently, MathTutorBench and GuideEval argued for evaluating open-ended pedagogical capability and adaptive instructional guidance directly rather than inferring them from generic helpfulness [redacted].
item_1883: Most interventions stem from repetition, low reliability, or generic low-yield turns, showing rectification targets long-horizon drift rather than format errors alone.
item_1935: However, the primary goal of these systems is often to study emergent behaviors rather than to optimize the quality of any single generated review.
item_1819: Temporal, audio-visual, frame-budget, and position probes

Temporal probes test whether the model uses event order rather than treating media as unordered feature bags.
item_1829: Nothing else: no VWA-reddit cell and no B2 cell carries a replicate, so every band below is imported into those cells rather than measured in them.
item_2053: These typically involved subtle reasoning differences where harm was implicit rather than explicit-for example, cases where the distilled model's rationale normalized a problematic premise through matter-of-fact discussion without explicit harmful language.
item_1849: None of the methods reduces the underlying failure, which is set by the base model's behavior on paginated search rather than by any adaptation.
item_2109: Beyond formal domains, recent work extends step-level evaluation: REVEAL [redacted] collects step-level human annotations distinguishing attribution and logical correctness against evidence, but labels classify error presence rather than explicit type, and evidence is retrieved post-hoc rather than given to the model as input.
item_2027: Because Layers 1 and 2 filter independently rather than by majority vote, denser poisoning cannot tip the defense the way it could tip a voting aggregator.
item_1856: Full filing texts are stored as
retrieved and unredacted, and this is counted rather than assumed: 758 of the 1,123 plain-text
documents (67.5) carry an /s/ signature block with a name, so most name at least one officer,
director or counsel.
item_1274: In the end, we had better results with the simpler, larger model rather than with the more complex, smaller model.
item_1995: Prior work typically focuses on conceptual tactics (e.g., emphasize mutual interest), rather than
actionable tactics in a specific negotiation scenario (e.g., politely decline to lower the price, but

2

We use `2 -regularized Logistic Regression classifiers.
item_2031: The labels here indicate the intended illustrative function rather than an absolute classification.
item_1949: We find, furthermore, that the correlation between irregularity and frequency is much
more robust when irregularity is considered as a
property of whole lexemes (or stems/paradigms)
rather than as a property of individual word forms.
item_1816: MEMORA-Planning Results

Planning tests whether formed memory can guide future action rather than only answer retrospective questions.
item_1222: The goal of these works is to learn
word-level POS tags, rather than sentence-level
syntactic embeddings.
item_1964: 4.3

Model
JAS
DRLM-Cond
Bi-LSTM-CRF
CRF-ASN
SelfAtt-CRF
DAH-CRF + MANUALconv
DAH-CRF + LDAconv
DAH-CRF + LDAutt
Human Agreement

SWDA
71.2
77.0†
79.2†
80.8†
82.9†
80.9
80.7
82.3
84.0

MRDA
81.3
88.4
90.9†
91.4†
91.1†
91.2
92.2
-

DyDA
75.9
81.1
83.6
86.5
86.4
88.1
-

Table 2: DA classification accuracy. † indicates the results which are reported from the prior publications.

topic) for DAH-CRF model training rather than
the topic labels automatically acquired from LDA;
DAH-CRF+LDAconv : Use conversation-level
topic labels automatically acquired from LDA for
DAH-CRF model training.
item_1886: Memorization Robustness

A first concern with any benchmark over popular novels is that the headline gain reflects pretraining memorization of the source text rather than the targeted construct.
item_1249: That the direct possessive sense is ruled
out here follows from the fact that the whole dialogue happens on the speaker’s property, so the
rope belongs to him, rather than to the addressee.
item_2090: Furthermore, DSPy fails Experience Persistence: it treats experience utilization as a discrete compilation process rather than a continuous memory accumulation.
item_1994: Thus, rather than start from a taxonomy of discourse relations like that used in PDTB, we characterize the different kinds of inferential relationships involved in interpreting imagery separately.
(a) T EXT: Cover with a sin- (b) T EXT: Let cool 5 minutes
gle layer of ravioli.
before spooning onto individual plates.

• To characterize temporal relationships between imagery and text, we ask if the image
gives information about the preparation, execution or results of the accompanying step.
• To characterize the logical relationship of imagery to text, we ask if the image shows one
of several actions described in the text, and if
it depicts an action that needs to be repeated.
• To characterize the significance of incidental detail, we ask a range of further questions
(some relevant specifically to our domain of
instructions), asking about what the image
depicts from the text, what it leaves out from
the text, and what it adds to the text.
item_2081: White, Black, Asian, Hispanic and Latino) as they appear in the studied datasets, rather than attempting to model the full sociological complexity of these constructs.
item_1978: Compared with these methods, our adversarial method
samples adversarial examples from the real-world
data rather than generating pseudo noisy perturbations.
item_1976: Rather than obtaining the sentimental inclination
of the entire text, ATSA instead aims to extract
the sentimental expression w.r.t. a target entity.
item_1967: They posit lower risk from
coverage gaps for machine learning work on predictive modeling, commenting, “since the purpose of this kind of machine learning research is to make inferences about out-ofsample observations rather than to test hypotheses about a
population, such research may be less sensitive to variation
due to missing data.”

2

The r/SuicideWatch subreddit, https://www.
reddit.com/r/SuicideWatch/, is a forum providing
“peer support for anyone struggling with suicidal thoughts,
or worried about someone who may be at risk”.
item_2073: This ensures the evaluation targets parametric knowledge retrieval rather than multi-hop reasoning capabilities, aligning with our goal of probing atomic belief states.
item_1273: The first one is that
we randomly choose n-grams to mask in the input
text stream rather than BPE tokens, and the second
one is that we predict the translation candidates of
a source n-gram rather than predicting the source
n-gram itself.
item_1746: Creativity and Originality

Unique Perspective : Does the speech reflect the speaker's creativity or unique perspective, rather than relying entirely on conventional templates?
item_1910: Exploiting the Brier Score, we compute the Non Conformity Score (NCS) that measures the variability in the model's confidence when predictions are compared with human annotation [redacted] The NCS penalizes the model both for assigning low probability to the 
reference label [redacted] and for placing the remaining probability on a single 
wrong label rather than spreading it across several.
item_1778: This balancing is approximate rather than strict: if one group does not contain enough eligible A/B candidates, the remaining quota is filled from the rest of the available Tier A/B pool.
item_2074: Because some challenge types apply only to certain tasks, the design is balanced within each task type rather than fully crossed.
item_2075: Because our evaluation relies on observed actions, our conclusions are behavioral rather than mechanistic, and reported metrics should be interpreted as behavioral adaptation rather than literal mental-state inference.
item_2126: Common measurements largely reflect exposure-related priors—such as corpus size, retrieved-document language ratios, gold availability, and cultural prior—rather than the genuine preference of the retriever or generator [redacted].
item_1743: This question is closely related to the idea of compositional generalization, where models are expected to combine concepts and contextual information rather than rely on dominant learned associations [redacted].
item_2024: This marks a shift toward content-adaptive encoding, where the computational budget is dynamically allocated to high-value signals rather than wasted on uniform processing.
item_1256: We believe that the higher percentage of newly
incorrect predictions on the rewritten development
sets demonstrates the brittleness of the NLI system rather than semantic dissimilarity that may be
introduced by the rewriter.
item_1726: The judge has access to the oracle answer for scoring, but the feedback text identifies where the reasoning went wrong rather than what the correct answer is.
item_1983: They usually focus
on depicting broad poetic images rather than details.
item_1926: This raises the possibility of a morphological shortcut, in which predictions are driven by morphology rather than specific factual knowledge.
item_1248: So focusing more on these words
according to their importance would give better results rather than focusing on all words.The key difference to other neural networks is that our system
focuses on the importance of headline for political bias detection in an article and discover which

sequence of tokens are relevant rather than simply filtering out.
item_2064: Making such a 
per-step policy train-free, rather than relying on fine-tuning as current 
approaches do, is a natural direction for future work.
item_1782: Corpus size and annotation quality likely contribute to the absolute numbers, and we therefore read these results as evidence about corpora as they exist rather than about coverage in isolation.
item_2056: This justifies treating (A1) as an idealization rather than a literal claim: the exploration-gain term in Theorem [redacted] is expected to shrink by a factor of [redacted] under correlation , and the observed correlation is bounded well below .
item_1761: As a result, this paper should be interpreted primarily as an empirical characterisation of model behaviour rather than as a prescription for improving MIP reliability.
item_1846: Cross-Attack Transfer

Cells are indexed by attack strategy rather than topical surface form ([redacted]); we test the strong property that a cell shaped by one jailbreak method also defends against others sharing the underlying strategy.
item_2015: A related form, the “potential deobjective,” expresses disposition of an agent rather than a real
action; however, this semantic distinction is not relevant to this analysis.
(2)

a.

(3)

Sake a-ku
sake 1 . -drink
‘I drink sake.’ [ain]
b.

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
item_2033: This results in a connected narrative rather than a list of disjointed statements, maintaining the logical progression and contextual flow needed for the next step.
item_1889: Top-1 accuracy collapses soft
orientation distributions into hard labels, expected calibration error depends
on binning choices, and generation-control diversity or control-realization
metrics are descriptive system analyses rather than direct measures of
empathy, naturalness, or human preference.
item_2093: Controlled Acoustic-Anchor Ablation
  
  To verify that the acoustic anchor contributes speech-grounded information rather than merely additional model capacity, we keep the complete SAMA-ASR architecture and trainable parameter count fixed while corrupting only the acoustic-anchor input.
item_2035: These results confirm that the two core phenomena identified in our analysis are robust across model families and evaluation tasks, rather than artifacts of a specific architecture or benchmark.
item_2023: The figure shows that the routing distribution changes substantially as training proceeds, rather than staying fixed around a preset ratio.
item_1722: It flows smoothly, uses warm and context-appropriate language, integrates guidance naturally, and feels like a thoughtful supportive reply rather than a scripted or encyclopedic answer.
item_1987: We proposed an approach to align the IndoWordNet, which is the ﬁrst lexical resource in Indian languages, with the PWN by taking advantage
of existing linkages between the IWN and the PWN
synsets.However, rather than focusing on the lexicalization problems and polysemy in IWN, we gave
full attention to map one synset from IWN to one
concept in UKC.
item_2006: We demonstrate that a system trained on
combined data achieves better predictive performance when experts annotate difficult examples
rather than instances selected at i.i.d. random.
item_2014: In our
case, however, rather than collecting cities with a
population of at least 100K, we consider all towns
with a population of at least 15K.
item_2072: Across all 12 systems, per-turn conciseness scores average [redacted] (std ), and under [redacted] of conversations fall below [redacted] ([redacted]), confirming that in practice the threshold screens only extreme verbosity rather than typical performance.
item_2087: MolmoPoint uses a dedicated pointing architecture that selects image locations from visual features, rather than generating coordinates as text.
item_2129: We further evaluate PersonaPlex [redacted], a full-duplex model with explicit role control, which reaches 0.06 GA: despite that control, it inverts the two roles, greeting the user and offering services rather than pursuing its goal (see Appendix [redacted]).
item_1725: While conceptually aligned with our goals, Safe RLHF operates at the RLHF stage rather than addressing the specific challenge of difference-awareness, where "safe" behavior (denying all group differences) directly contradicts task accuracy.

[redacted] propose iterative constitutional alignment, progressively refining model behavior through multiple rounds of principle-guided self-improvement.
item_1992: Unlike
most current methods, conceptor debiasing uses a
40

Proceedings of the 1st Workshop on Gender Bias in Natural Language Processing, pages 40–48
Florence, Italy, August 2, 2019. c 2019 Association for Computational Linguistics

soft, rather than a hard projection.
item_2057: Released traces preserve tool
definitions, tool-call arguments, and tool outputs as structured JSON
fields rather than flattened text, so scaffold-aware defenses (boundary
enforcement, schema validation, tool-output sandboxing) can be
evaluated against the same case set without re-running models.
item_1834: MEMORA thus maintains memory selectively rather than storing every observation as an independent record; Figure [redacted] quantifies a median [redacted] reduction relative to the unedited observation stream on the 18-participant EPIC-Kitchens corpus (state histories preserved).

*[t]
Active memory formation in Entity Memory (18 participants).
(a) Cumulative records with and without the editor (per-segment median, first 150 segments).
(b) Editor decision mix (, P01-P04): Noop () and Upd () dominate Add () with rare Del () - the editor mostly maintains rather than expands.
(c) Per-participant size before vs. after editing; median reduction , minimum .
item_2107: This confirms that the improvements stem from state-conditioned allocation of lookahead, rather than stochasticity or horizon variability per step.
item_1880: 4-Step Completeness is an unambiguous structural check, so sharing its simple form between reward and metric reflects clarity rather than circular validation.
item_2112: We therefore interpret AMD as a construct-aligned proxy rather than a direct observation of interlocutors' latent affective meanings; detailed estimation-pipeline limitations are discussed in the Limitations section.
item_1286: Current research in this field is focused on detecting and
correcting for gender bias in existing machine
learning models rather than approaching the
issue at the dataset level.
item_1745: Residual correction (RC) proves essential rather than optional: removing RC while retaining LDP (=2.0) yields MAE worse than the single-LLM baseline across all datasets, with the UCI dataset showing the most dramatic degradation (w/o RC: 0.2997 vs.
item_1239: This
is enabled by our model designed to embed
attribute contextually rather than attribute tag
along.
item_1985: In this
case, the term Multinomial Naı̈ve Bayes lets us
know that each p(fi |c) (where fi is a feature and
c the category or the class) is a multinomial distribution, rather than some other distribution such as
a Bernoulli distribution.
item_1893: Phase VII: Authority-Weighted Terminal Audit

The terminal audit (Judge-Auditor) reaches a final, citable verdict, resolving cross-modal disagreement via claim-type-conditioned authority rather than the drafting model's own judgment.
item_1948: But from a pedagogy perspective, content
from these systems may be inappropriate - for instance, the questions generated are often factual
rather than encouraging critical thinking (Rickford, 2001).
item_1744: No model clears 26 on any metric except Task 3 metadata EM, which [redacted] shows is inflated by constant and null fields rather than extraction.
item_2111: Training uses GRPO (Group Relative Policy Optimization) [redacted] with our dense reward signals from code instrumentation, enabling the agent to receive partial credit for intermediate progress rather than binary success/failure.
item_2147: We analyze 10 MemGuides evolved on FutureX and categorize each guide by its overall evidence-processing pipeline rather than by the surface wording of the guide.
item_2130: The client's evolving trust gates disclosure of the profile's content, and resistance spans multiple clinical dimensions rather than a single label.
item_2121: Formally, given a dataset [redacted] that needs to be unlearned , we frame unlearning as a negative alignment of preference over a pair of policies:






In contrast, for conventional alignment methods such as DPO, the preference is applied to pairs of data samples rather than policies (Equation [redacted]).
item_1276: We can speculate that top layers focus on
predicting the future rather than incorporating the
past, and, at that stage, token frequency of the last
observed token becomes less important.
item_1267: Rather than a re-ranking frame-

BiLSTM Layer

The core of this base model is a bidirectional
recurrent neural network, in particular a Long
Short-Term Memory neural network (Graves and
Schmidhuber, 2005).
item_2116: To suppress spurious state transitions, an online model-based estimator [redacted] is triggered conservatively in stable batches of [redacted] turns rather than at every execution step.
item_2062: Out-of-Distribution Generalization

To ensure that identifies universal causal mechanisms of toxicity rather than overfitting to dataset-specific artifacts, we evaluate the cross-domain transferability of our methods.
item_2044: Rather than aligning representations solely at the sentence level like prior MRL approaches [redacted], SIA focuses on token-level intra-relations derived from geometric similarity and attention-based importance.
item_2032: Community-summary retrieval surfaces
commentary about the topic rather than the cited primary source.
item_1873: The same frozen GPT-5.4 library reaches comparable OOD on two different executors, and in both cases the transfer exceeds the target's own self-trained library ( for Gemini, [redacted] for gpt-oss), so the gains track the source library rather than the executor.
item_1996: On the other hand, autoregressive training for
creating such content plans limits the model to
capture frequent sequence patterns rather than allowing diverse arrangements.
item_1776: We deliberately optimise the three concept definitions within each category for diversity rather than minimal contrast: we choose examples and mechanism phrasings that sit as far apart as is defensible within the concept, so that any residual rating differences we observe are driven by the conceptual core rather than by a single contestable wording choice.
item_1916: Position bias refers to a judge's tendency to favor a response based on its presentation order rather than its content.
item_2005: I think it’s a sampling bias rather than anyone massaging the numbers to see what they want to see.
item_2091: This saturation indicates a structural bottleneck: the "one-size-fits-all" parameterization suffers from gradient interference when optimizing for conflicting values, forcing convergence towards an averaged solution rather than distinct cultural modes.
item_2110: SFT rather than RL For Section [redacted], we acknowledge that Reinforcement Learning methods are currently the state-of-the-art for eliciting particular reasoning behavior, but we elect LoRA Supervised-Finetuning (SFT) due to resource constraints.
item_2021: It is tempting for an annotator who is not skilled at such tasks to only
glimpse through the long text, rather than read it
carefully.
item_1261: Here we report on a comparison of ELMo (Peters et al., 2018) and the Universal Sentence Encoder for English (USE) (Cer
et al., 2018) with two conventional word embedding methods, GloVe (Pennington et al., 2014) and
WTMF (Guo and Diab, 2012).4
ELMo is character-based rather than wordbased, relies on a many-layered bidirectional
LSTM, and incorporates word sequence (language
model) information.
item_1773: Fine-tuned DeBERTa checkpoints and Hugging Face model bundles
are released where permitted by the base-model and dataset terms.(https://huggingface.co/VictorYeste/value-context-rag-deberta-v3-base-doc-rag) For large instruction-tuned LLMs, we release only configurations,
prompts, and derived outputs rather than redistributing model weights.
item_1853: Blind pairwise human preference is treated as the primary evidence
for generation quality, while automatic metrics and internal
orientation-control diagnostics are used as supporting analyses rather than
substitutes for human judgment.
item_1754: Regarding Potential Risks of Systemic Bias During Annotation:
To mitigate potential annotator bias from LLM-provided candidate scores, our labeling pipeline treats model scores as non-binding references rather than ground truth.
item_1783: Consequently, models may succeed by exploiting low-level artifacts or static cues, rather than learning fine-grained action dynamics [redacted].
item_1765: However, the metadata ablation results are mixed rather than uniformly monotonic.

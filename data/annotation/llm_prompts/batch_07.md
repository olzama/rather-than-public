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
item_1980: The difference from ULF is that it focuses on binary structural relations
such as restrictor, body, or modifier between semantic components, rather than operator-operand type
structure.
item_1901: To address this issue, we apply a lightweight self-refinement strategy [redacted] for PoT and text2SQL (rather than a full-fledged LLM-based code debugging[redacted]).
item_1845: Because models emit an unconstrained coordinate rather than selecting among candidates, [redacted] is a forced-choice reference rather than the chance level of the prediction task, and a prediction can fall outside both candidate regions.
item_1876: For a fairness-and-deployment claim we want the population-averaged answer "how much less likely is a randomly selected clinician to attribute a full-condition vignette to AI?" rather than the conditional "how much less likely is participant [redacted] specifically?".
item_2137: We implement lookback-lens probes using sklearn's LinearRegression architecture rather than the paper's sklearn LogisticRegression architecture, as the LinearRegression sklearn implementation allows for training with soft target scores.
item_2004: Where there were multi-word expressions,
we took the average frequency of all words in the
multi-word expression, rather than taking the frequency of the N-gram.
item_1766: For rel-pos-horizontal, rel-pos-vertical, and proximity the target is selected on 0.89-0.90 of within-region predictions, indicating that the deficit is candidate localization rather than relational interpretation; the alignment estimate varies with the classification rule ([redacted]) and we do not interpret it.
item_1753: Subsequent studies emphasize temporal understanding as a particularly vulnerable dimension: models often confuse visually similar actions or infer event order from language priors rather than observed motion [redacted].
item_2089: Because optimization
is now distributed across [redacted] reveal-count levels rather than
concentrated on the fully masked state, we enlarge the optimization
budget relative to the single-state baseline-fourfold for LLaDA and
twofold for Dream-so that each level receives sufficient
optimization; for Dream, we also loosen the norm clamp on
 accordingly.
item_1250: 2002), or perform
comparisons at stem and synonymy levels, rather than exact match only, namely, Meteor
(Banerjee and Lavie 2005).
item_1959: Although
many models represent text better (e.g. sequence
models, tree models, etc...) we limit ourselves to a
simpler model to show the improvement by the domain adaptation technique rather than by the text
model.
item_1971: The
chopsticks are used to move food to the Agent’s
mouth rather than eating the chicken.
item_1993: In these cases, we assume that
humans produce captions by distributing their attention more or less evenly across all image regions rather than focusing on a small number of
highly important regions.
item_2104: We find
that (i) ASR errors degrade multi-hop QA performance, with
degradation scaling with WER; (ii) structurally complex
retrieval methods amplify rather than absorb these errors;
(iii) query-entity corruption is the dominant failure mode; and (iv) lightweight surface-form mitigations close
only a small fraction of the gap.
item_2119: Since this model functions as a low-competence baseline rather than a primary test case, we set , which substantially exceeds the thresholds derived from the two higher-competence models while keeping experiment counts manageable.
item_2108: Rather than retraining models for each domain, we represent the anonymizer as an LLM guided by a natural-language instruction prompt .
item_1988: Introduction

Over the last decade people tend to search
for products online rather than physically on
stores.
item_1268: Software Support MRP scoring is implemented
in the open-source mtool software (the Swiss
Army Knife of Meaning Representation), which is
hosted in a public Microsoft GitHub repository to
stimulate community engagement.8 mtool implements a refinement of the maximum common edge
subgraph (MCES) algorithm by McGregor (1982),
initializing and scheduling candidate node-to-node
correspondences based on pre-computed per-node
rewards and upper bounds on adjacent edge correspondences.9 In addition to the cross-framework
MRP metric, the tool also provides reference implementations of the SDP, EDM, SMATCH, and
UCCA metrics, in the case of SDP and UCCA
generalized to support character-based anchoring
(rather than using token indices).
item_2114: Black-box Compatibility
A key advantage of this approach is that it relies entirely on prompt-routing and context manipulation rather than internal parameter updates.
item_2115: However, as our earlier analysis suggests, general abilities are governed by mappings between directions rather than individual parameters.
item_2041: A frequent failure mode in ETD is that models default to surface-level atomic verbs (e.g., "walk," "look," "sit") rather than identifying the intended narrative-level predicate (e.g., "confront," "betray," "reconcile").
item_1937: We list representative tools exposed to the target agent across five environments, focusing on core functionalities rather than exhaustive definitions.
item_1724: Our setting differs because the retrieved evidence is fused in parameter space rather than prompt space: each passage induces a candidate parametric update, and the core question becomes how strongly each update should influence the final merged adapter.
item_1962: More
work is needed to explore this trend further, and

10

We experimented with multiple variations on this
mapping, including using the z -normalized (rather than the
raw) human scores, and using bins based on percentiles
rather than evenly spaced over the full range.
item_1915: Across all language pairs, and exhibit consistently lower correlation with other top-performing WMT24 metrics, suggesting that [redacted] may provide a distinct evaluation signal rather than mirroring existing metrics.
item_1229: Because the relations
between parent and child categories in Wikipedia
do not strictly correspond to IS-A relations, it
would be more correct to consider the scores for
this source as measures of semantic relatedness
rather than semantic similarity.
item_1226: However, sometimes we would like to express
the fact that a certain lexical entry evokes a certain mental
concept rather than that it refers to a class with a formal interpretation in some model.
item_1740: The focus is on appreciating the map as an artistic object rather than seeking the destination.
item_1805: Ablations localize it to the validation gate rather than the skill-writing process, which alone closes none of the gap against the no-skills baseline.
item_1843: This limit belongs to today's agents rather than to routing itself.
item_2008: The two other
discourse relations, on the other hand, are supported by the semantic,
rather than the structural relations between their components.
item_1791: Rather than relying on multi-step
pipelines or inference-time voting, we constrain the editing scope natively through GEC taxonomy-based instructions.
item_1911: Per-task centroid distances are consistently lower than the global setting, confirming that over-refusal is a within-cluster perturbation rather than a cross-task signal.
item_2066: Because we utilize a smaller, cost-effective model (GPT-5-mini) and perform comparisons based on RAG-retrieved content (i.e., comparing relevant retrieved text chunks rather than feeding the entire full texts into the context window), the average cost is only 0.075 per paper according to OpenAI's API pricing.
item_2003: In a few cases, it appears to be slightly better
to train directly on non-projective trees rather than
on optimally projectivized trees.
item_1257: Also, we would
not expect frequency to be a good measure in other
contexts, such as how powerfully an entity is portrayed in a single document rather than across a
large media corpus.
item_1924: A natural question is whether the ensemble gains simply reflect a larger retrieval budget rather than true complementarity.
item_1914: This approach emphasizes globally salient, language-skewed neurons rather than fine-grained feature sharing.
item_2055: Evidence Attention Mass measures total evidence alignment, Pointing Accuracy measures whether the strongest attention peak lands on evidence, and Multi-evidence Coverage measures whether attention is distributed across multiple supporting regions rather than concentrated on a single salient region.
item_2000: The model deviates from existing graph models as it focuses on constructing unique nodes
and edges, encoding information into edge representations rather than node representations.
• The proposed model is independent of syntactic dependency tools and can achieve stateof-the-art performance on a manually annotated, document-level chemical-disease interaction dataset.
• Analysis of the model components indicates
that the document-level graph can effectively
encode document-level dependencies.
item_1730: This result reinforces our finding that dynamic retrieval heads originate from a broad, long-tail distribution of attention heads rather than being confined to a slightly expanded static subset.
item_1263: We only look at the impact of using an
LSTM cell rather than a recurrent cell since it was
a better technique across the board (see previous
section).
item_1728: Further, we test artificial neural networks rather than humans.
item_1866: First, the critic is framed as a recall-preserving reranker rather than a hard blocker; when mechanism evidence is incomplete or ambiguous it is instructed to return maybe rather than doesnotapply, deferring the close call to the guard so that recall is not collapsed on borderline cells.
item_1942: This is consistent with the general finding
in structural-probing literature that the latest layers of
language models tend to be optimised for the prediction
objective rather than for representational explicitness, and
that intermediate layers often carry more directly recoverable
structural information [redacted].
item_1282: This implies that k is more sensitive to the
target task rather than the target language, which
we discuss further in § 5.3.
item_1812: 3 &
  Adds anhedonia and study/social impairment; patient denies elevated-mood episodes and psychotic experiences; mentions occasional chest tightness. &
  Anxiety frequency, dominance, severity anchors, safety risk, and somatic disambiguation. (5 open) &
  Dep, Anx, Mix (Oth removed: duration exceeds the 2-week threshold) &
  Criterion entries on quantifying anxiety episodes and weighing them against depressive core symptoms; used to plan the next inquiry rather than to answer it. &
  4/5; exclusion answers are internally consistent; chest tightness deferred to a later somatic check.
item_1786: All 243 NEI training
examples have empty evidence fields, making NEI trivially
detectable from evidence absence rather than genuine
reasoning.
item_2050: In practice, meaningful improvements require deliberate, language-aware interventions rather than purely language-agnostic scaling.
item_1808: Context Today is datetime. agentpersonality

Specific Instructions and Policies: agentinstructions

 Voice-Friendly Communication Rules

 Natural Speech Patterns

    Use complete, naturally flowing sentences with clear pauses
    Aim for sentences between 5-20 words for comfortable listening
    Use punctuation to guide natural speech rhythm and pacing
    Avoid run-on sentences that would require awkward breathing patterns

 Clarity When Spoken Aloud

    Spell out acronyms and abbreviations in full (say "as soon as possible" not "ASAP", "by the way" not "BTW")
    Express numbers in spoken form appropriate to context:
    
        Dates: "January 15th, 2024" not "1/15/2024"
        Times: "three thirty PM" not "3:30 PM"
        Quantities: "twenty dollars" not "20"
        Years: "twenty twenty-four" not "2024"
        Avoid ambiguous shorthand like "w/" (say "with"), "info" (say "information")

 Audio-Appropriate Content

    Never use visual-only elements (tables, bullet points, formatted lists, URLs)
    Convert structured information into conversational summaries
    Describe rather than display (say "I found three options" then list them naturally)
    Skip content that only makes sense visually (links, email addresses, code)

 Prohibited Elements

    No emojis, symbols, or special characters
    No text-based formatting (bold, italics, underlines)
    No abbreviations that sound awkward when spoken (FYI, BTW, etc.)
    No visual shortcuts like "" (say "and"), "+" (say "plus")

 Conversational Behavior

 Response Style

    Keep responses brief and conversational (2-4 sentences typically)
    Summarize long lists rather than reading them exhaustively
    Use natural transitions between topics
    Maintain a warm, professional phone conversation tone
    Avoid overwhelming the user with too much information at once.

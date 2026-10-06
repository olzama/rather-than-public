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
item_1770: The gate is highest when notes describe active clinical change, consistent with text acting as an observation channel for latent dynamics rather than a static side input.
item_2125: MovieGraphs [redacted] annotates human-centric situations in movie clips with interaction graphs but operates on short scenes rather than full-length narrative arcs.
item_1954: To our knowledge, this is the first algorithm to accomplish this for input-output mappings rather than phonotactics.
item_2061: However, the negative wording critiques analytical depth rather than declaring the evaluation invalid, and the positive assessment is explicitly hedged by the phrase “as far as I can tell,” indicating limited confidence and leaving room for unobserved weaknesses.
item_1966: Constrained decoding in MT
(Post and Vilar, 2018, i.a.) has been used to
enforce the use of specific words in the output,
rather than constraints on tree structures.
item_1811: Hybrid systems are Audio-Native on the input side while retaining a text-to-speech output stage, and occupy a middle ground between fully cascaded (STTLLMTTS) and fully end-to-end (S2S) pipelines.      
  [Audio-Native.] An umbrella term for voice agent architectures that process audio directly at one or more stages, rather than relying solely on text-based STTLLMTTS cascades.
item_2063: Paradigms that depend
on more specific verbal behavior, such as control, raising, passivization,
causative-inchoative alternations, and object drop, use curated verb lists
rather than unconstrained frequency-filtered sampling.
item_1255: The main idea
is to find out the most suitable words for

Figure 2: The hypernym-based translation between Princeton WordNet and Bilingual dictionary
on a given word “chemist”
the concept in terms of linguistic context use
rather than word-for-word translation between
synsets.
item_1908: This suggests that the heads implement knowledge-based matching rather than pure gating.
item_1953: Recommendation as a Communication Game:
Self-Supervised Bot-Play for Goal-oriented Dialogue
Dongyeop Kang♥ Anusha Balakrishnan♣ Pararth Shah♣
Paul Crook♣ Y-Lan Boureau♣ Jason Weston♣
♥
Carnegie Mellon University, ♣ Facebook AI
[email] [email]
{pararths,pacrook,ylan,jase}@fb.com

Abstract
Traditional recommendation systems produce
static rather than interactive recommendations
invariant to a user’s specific requests, clarifications, or current mood, and can suffer from
the cold-start problem if their tastes are unknown.
item_2059: Because word order correlates with other language properties, the observed SVO advantage may also be an artefact of a confound such as morphological complexity [redacted] rather than a genuine word order preference.
item_2103: As users increasingly rely on these synthesized overviews rather than navigating to original sources, the design choices embedded in generative search systems, i.e., which sources they retrieve, how they synthesize them, and what language they use, can have great impact on how they shape public information exposure [redacted].
item_2016: In other words, with
the duality constraint, Gθqr tends to generate diverse responses rather than safe responses.
item_2012: Since the focus of our work is on structural
rather than lexical simplification, we follow the
approach taken in Sulem et al. (2018c) in terms
of S IMPLICITY and restrict our analysis to the
syntactic complexity of the resulting sentences,
which is measured on a scale that ranges from
-2 to 2 in accordance with Nisioi et al. (2017),
while neglecting the lexical simplicity of the output sentences.
item_2065: Concept Spaces and Concept-Based Interpretability

Concept-Based Interpretability
Concept-based interpretability explains model behavior through human-interpretable concepts rather than opaque latent features [redacted].
item_2098: Summarized in Table [redacted], most benchmarks remain static: they evaluate QA over a pre-built history rather than memories from real interactions.
item_1852: In contrast, we focus on full-story generation rather than next-chapter prediction, and treat thinking as a means for better stories rather than as the primary object of optimization.
item_2133: We cover [redacted] and keep a
fact at length [redacted] only if its object tokenizes to exactly [redacted] tokens
under both the LLaDA and LLaMA tokenizers, so that any
ARM-MDM difference reflects target length rather than one tokenizer
splitting the same target more finely than the other.
item_1888: Those moments were preserved in sharp pieces rather than a narrative.
item_1727: 1GLM entered the forced search tool-call format but degenerated into token repetition until exhausting the generation budget, on about 15 of examples, at both the original and a doubled budget; we therefore report this configuration as unable to sustain the search loop rather than as an accuracy number.
item_1943: AST sequence similarity parses each tool into a Python AST, records the sequence of AST node types visited by ast.walk, and compares these node-type sequences with the same sequence-matching ratio; it focuses on structural similarity rather than exact lexical overlap.
item_2120: Since all agents observe identical but incomplete evidence, differences in their conclusions arise from reasoning rather than information asymmetry.
item_2085: While some systems also employ a general practitioner, triage in most EDs is performed exclusively by nurses, who follow structured protocols rather than diagnostic reasoning.
item_2080: Unlike standard task-focused reasoning benchmarks in domains such as mathematics, coding, or QA, TimeBench is designed to isolate reasoning under temporal discontinuity, anomaly, and contextual re-anchoring rather than temporal fact recall or time-sensitive knowledge retrieval.
item_1741: To test the presence of priming, we analyse only the stimuli where the prime and target sentence are not semantically coherent (the unrelated condition), so that any increase in structure-matching completions can be attributed to structural priming rather than semantic facilitation.
item_1253: More specifically, they argue that rather than understanding those implicit rules and being able to
compositionally apply them, RNN models exploit
biases in the data that are unrelated to the underlying system.
item_1898: While we do not explicitly evaluate VAUQ in these settings, the core principles of vision-aware uncertainty quantification remain applicable, though extending the framework may require modeling how visual information contributes across multiple reasoning steps rather than only at the final response level.
item_1844: Creative-writing assistants may require alignment objectives distinct from general-purpose assistants: domain-conditional reward models that calibrate to the source register rather than a pooled human preference; distributional matching objectives that preserve variation within a continuation as well as across stories, rather than only point-wise quality; or preference data deliberately sampled from the literary tail rather than from majority-preferred continuations.
item_1896: Because relies on a writable memory store rather than fixed weights, poisoning resistance is evaluated directly rather than assumed.
[redacted] stress-tests a black-box attacker who controls the feedback loop and a white-box attacker who writes crafted cells into the store, reporting retrieval and critic-filter rates alongside ASR/FRR, and shows that provenance-tied authority closes the white-box channel.
item_2140: Because the answer is a single multiple-choice label rather than a free-form span, all methods are far more robust to compression here than on longbookqaeng: every method stays within a few points of the Full KV baseline () across all three ratios.
item_1882: This is deliberate: prose is the sole authoritative source for causal attribution here, so a single well-grounded passage is treated as sufficient coverage rather than automatically contested, and the terminal audit (Section [redacted]) still independently checks the drafted sentence against that same evidence.
item_2047: Predicted scores often cluster around specific values rather than varying smoothly.
item_1979: This phenomenon is even
more prominent in ST-NT alignment (the
alignment of the ST with the notes that were taken
during the listening phase) since notes are a byproduct of ST understanding and a predecessor of
TT production, rather than a shorthand of neither
the ST nor the TT.
item_2071: Conclusion


Our study reveals that widely used open-ended generation metrics capture Surface Compliance rather than genuine parametric reconfiguration.
item_2069: The policy, reward judge, and recovery operator all consume [redacted] rather than raw history.
item_1963: 5.2

Multi-Granularity Network

We propose a model that can drive the highergranularity task (FLC) on the basis of the lowergranularity information (SLC), rather than simply
using low-granularity information directly.
item_2013: As a result, the
algorithm produced the lemma *эмиратла
and the MSD *Case=Acc of the NOUN эмиратланы rather than эмират ‘emirate’ and
Case=Acc, Number=Plur.
item_2095: This characterizes the recall achievable under idealized guidance, serving as an upper bound rather than an estimate of real-world human gains.
item_2022: Model identity is moderately associated with response distribution on every layer ( to , all [redacted] ), so this variation is systematic rather than noise.
item_1909: Only by integrating the intervening Steps 2 and 3 can the model infer that the current observation aligns with Step 4 (green) rather than Step 1 (purple).
item_1982: Computational methods for fighting fake
news mainly focus on automatic fact-checking
rather than looking at writing styles of news
articles (Potthast et al., 2018).
item_1279: The Effect of Translationese in Machine Translation Test Sets
Mike Zhang
Information Science Programme
University of Groningen
The Netherlands
[email]

Abstract

rather than the opposite (Kurokawa et al., 2009;
Lembersky, 2013).
item_2123: Limitations

This work focuses on evaluating whether LLM-based monitors can detect the presence of misbehavior, rather than precisely localizing or attributing it within long solutions or reasoning trajectories.
item_1989: Rather than labeling the disfluencies in the original target data,
Turkers were asked to rewrite the utterance in a
‘copy-edited’ manner without disfluent phenomena.
item_1792: The survey prioritizes breadth and diversity over completeness, selecting representative resources to reflect methodological trends, task coverage, and language diversity rather than listing every available work.
item_1998: We also exclude
marketing strings such as ‘What’s New’ that typically require transcreation, adaptation, or more
idiomatic rather than literal translation.
item_2145: We filter out all samples for which [redacted] is answered incorrectly, ensuring that the reported values reflect base confidence rather than raw accuracy.
item_2092: It is sufficient for the present paper’s purpose, but limited in construction scope: it contains 77 scenarios, covers only a finite set of diagnostic patterns, and was developed alongside the framework rather than independently.
item_2009: Also,

the mixed nature of such texts suggests an abstractive, rather than extractive study.
item_1233: This observation is intuitive
in the sense that we would expect to generally get
better privacy protection when any single personal
clinical note is mixed with more, rather than fewer,
notes in the train-set of a note-generating model.

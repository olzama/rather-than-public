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
item_1750: First, by examining individual countries rather than Latin America as a whole, EspanStereo captures more culturally specific target groups.
item_2139: These permutations are controlled reconstruction stimuli rather than intended natural variants: they preserve the same four constituent units while disrupting their canonical internal order and retaining a unique gold target.
item_1923: With [redacted] agent-native
models against [redacted] general, inferential statistics are underpowered;
we report effect direction and magnitude rather than -values, and we
treat the gap as an effect-size estimate rather than a confirmed
population claim.
item_1799: The taxonomy is curated rather than exhaustive: it targets cross-modal evidence assembly (word identity/order, dense visual signal, audio presence) and excludes trivially recoverable perturbations; per-operator definitions, severity parameterisations, and rejection criteria are in Appendix [redacted] (Table [redacted]).
item_1219: The model training is unsupervised, relying on only the speaker’s conversation history rather than meta information (e.g.,
age, gender) or audio signals which may not be
available in a privacy-sensitive situation.
item_2051: We adopt a similar reverse-direction intuition, but apply it to conditional perplexity for training data selection rather than post-hoc evaluation.
item_2038: Building on this, we advocate extending disagreement-aware and annotator-aware frameworks toward diversity-aware modeling that explicitly accounts for culturally patterned differences in what counts as an empathy need, rather than collapsing them into a single consensus label.
item_1950: Bernardy et al. (2018) suggest that adding context causes
speakers to focus on broader semantic and pragmatic issues of discourse coherence, rather than simply
judging syntactic well formedness (measured as naturalness) when a sentence is considered in isolation.
item_1936: We apply this constraint to social measurement: extracting a valid construct () from text requires actively modeling and neutralizing nuisance factors (), rather than assuming that an unsupervised embedding will spontaneously isolate them.
item_1981: In addition, rather than compute IDF weights w(e) and

|E| + 1
)
|E∃e | + 1
|F| + 1
)
w (f ) = idf (f ) = log(1 +
|F∃f | + 1
P
max w (e) · s (e, f )
e∈e f ∈f
P
precision =
w (e)
e∈e
P
max w (f ) · s (e, f )
f ∈f e∈e
P
recall =
w (f )
w (e) = idf (e) = log(1 +

f ∈f

YiSi-2 =

2 · precision · recall
precision + recall

where s(e, f ) is the cosine similarity of the vector representations v(e) and v(f ) in the bilingual
embeddings model.
item_1885: The fact that training-free interventions move the needle only on the composition-side errors, and only when the intervention bypasses the relation (SoM) rather than scaffolding it (CoT, steering), is consistent with the discussion section's reading that the bottleneck is on the perception / binding side, not on the language side.
item_1760: Evidence-based prompting offers limited benefit here, as such misbehavior may typically diffuse and be distributed across the context rather than localized in a single identifiable span.
item_1245: We presented evidence that simple neural networks
for satire detection learn to recognize characteristics of publication sources rather than satire and
proposed a model that uses adversarial training to
control for this effect.
item_2020: As the
speech becomes more natural, it becomes harder
to obtain large manual transcripts for a large portion of the data to carry out the studies that we
present, so we also validate an alternative method
for finding factors that influence the performance
of commercial systems, relying on agreement between systems rather than manual transcripts.
item_1931: Thus, residual reliability should be understood through aggregate spherical coherence rather than pairwise similarity.


*[t]



Main results on the Qwen3-4B-Base expert group across individual benchmarks and the overall average.
item_2079: Performance peaks around [redacted] and remains competitive for nearby values, suggesting that TabTrim only requires a moderate recall bias rather than precise tuning.
item_1281: Previous work showed that for the
one-to-many problem, conventional RNN-based
encoder-decoder models tend to generate generic
responses, rather than meaningful and specific answers (Li et al., 2016; Serban et al., 2016).
item_1997: Rather than working over words of a sentence,
given the formal nature of the proof, the projective
MST algorithm must work over symbols of the input word w.
item_1965: Most of
the generated conversations become off-goal dialogues with utterances being non-relevant or contradicted to goals rather than on-goal dialogues.
item_2082: Further analyses demonstrate that LMs fine-tuned on a single garden-path construction also capture human processing difficulty of other unseen garden-path constructions, but that the current method does not allow the models to explain processing difficulties that are likely accounted for by memory-based theories rather than surprisal theory (Section [redacted]).
item_1851: Because the sample is balanced by
construction, the evaluation set is near-parity (
eligible) rather than following the pool's base rate; F1 in
Table [redacted] should be read against that.
item_2036: Methodology

Empirical Motivation: Gradient Alignment Dynamics

The preceding analyses show that preventative steering is process-dependent: its protection is sustained by the ongoing steering signal rather than by a reusable weight change.
item_2048: Thus, their faithfulness, which concerns whether the identified components truly reflect the model’s internal mechanisms rather than artifacts of specific datasets or lexical distributions, remains unestablished.
item_2001: Compared to other text GANs with RL
training techniques, our framework acquires samples from the stationary distribution rather than the
generator’s distribution, and uses RAML training
paradigm to optimize the generator instead of policy gradient.
item_2002: Train
Valid
Test

Seqs

Tokens

Vocab

Avg Len
YC
KC

8,125
1,014
1,020

307,573
36,830
37,156

3,573
1,479
1,489

10.0
8.8
8.8

37.9
36.3
36.4

Table 1: K IDS C OOK corpus statistics

the action was performed, rather than hallucinating additional details.
item_1828: Introduction

Human label variation (HLV; [redacted]) highlights that annotator disagreement often captures meaningful differences in interpretation, rather than noise to be collapsed into a majority label [redacted].
item_2019: Though
TF enables efficient training, it results in “exposure bias” because agents must follow learned
rather than gold trajectories at test time.
item_1933: Because queries from one unit are not independent, we
also report a cluster (per-unit) bootstrap that resamples whole units rather than queries:
intervals widen, as expected, but every gain remains significant, and the smallest-domain headline
-Auto hybrid -survives even at its [redacted] units (cluster [redacted] CI ).
item_1232: The substitution characters are sampled uniformly for the alphabet, rather than attempting to
sample from a more informed distribution, which
has potential for further improvements.
item_1247: This might be the result of the tuning process of the sparse rational structure simply
learning a collection of words, rather than coherant phrases.
item_1785: Rather than using LM surprisal as one single model to capture human sentence processing, we should look to parts of LMs as models of sub-processes or components of human language processing.
item_2067: This increase is therefore part of the intended behavioral effect rather than an unrelated side effect.
item_1777: These are pipeline errors rather than agent errors, and filtering them deterministically before any judge invocation avoids wasting LLM calls on malformed simulations.

.
item_1758: However, MemRL stores LLM-summarized reflections rather than full structured execution traces, uses Q-value-weighted ranking rather than structural signature matching, and implicitly encodes success/failure through Q-values rather than explicit dual-outcome indexing.
item_2058: In this stage, we utilize the remaining training data to drive learning through semantic rewards rather than token-level supervision.
item_1918: In fact, we hope that a probabilistically calibrated sampling procedure such as large language Gibbs can help mitigate some of these issues by generating samples that are more consistent with an implicit underlying distribution, rather than introducing additional error from a non-iterative sampling procedure.
item_2099: In summary, we make the following contributions:
[leftmargin=*, itemsep=2pt]
    We propose Agentic Chain-of-Thought Steering (ACTS), a reasoner-agnostic framework that treats efficient reasoning as strategy-level steering over an evolving chain of thought, rather than global length control.
item_1855: This sweep was run on a later build whose canonical configuration omits the two optional agent-side switches, so the rungs should be read as relative increments rather than as a reproduction of [redacted]; the absolute canonical values there are the ones reported throughout the paper.
item_2011: In this approach
we use our model’s copy mechanism to copy tokens corresponding to the surface forms of entity
arguments, rather than copying entities directly.
item_1246: Additionally, rather than relying
on the availability of trained linguists to annotate
the corpus, our work explores how we can use
crowdsourcing to obtain hedge annotations.
item_1952: Their proposed LanguageModel-based Commonsense Reasoning (LMCR) will
give as more probable an instruction such as ”Pour the
water in the glass.” rather than ”Pour the water in the
plate.”.
item_1879: It therefore tests whether a model can identify a useful next evidence gap from public partial consultation context, rather than whether it can exploit simulator-specific response patterns.
item_2037: Each claim is resolved through Asymmetric Evidence Authority, which conditions evidence trust on claim type rather than treating all modalities as equally reliable; Chain-of-Custody Verification, which checks grounding at the hand-off between drafting and adversarial review rather than only at the pipeline's exit; an Adaptive Rebuttal Cycle, which routes contested claims through adversarial debate whose depth scales with what that debate finds; and a terminal entailment audit paired with a continuous Hallucination Risk Index that distinguishes claims that passed scrutiny from claims never contested.
item_2132: The physician audit establishes inter-physician consistency rather than automated-judge calibration.
item_1956: Hence, the number of sentences to be associated
with s2 is 25, rather than 125, thus maintaining
the distribution across senses balanced.
item_1991: We choose BERTBASE
in our work rather than another larger pre-trained
model BERTLARGE due to the resource limitations
and computation cost.
item_2077: Rather than searching directly for a single globally optimal set of alignments, we draw samples from the posterior over alignments using a Gibbs sampler that updates one word pair at a time:


    Initialize.
item_1815: To attribute such failures confidently to the agent rather than the user simulator, we must verify that the user's synthesized speech correctly conveys all critical entities.
item_1905: The dominance of wrong-event errors (78.0) suggests that models frequently fail to identify the correct narrative event at all, rather than merely selecting the wrong level of abstraction.
item_2046: The dominant error without it is judging whether the argument is positive about the topic rather than whether it agrees with the belief, which inverts the label whenever the belief is itself a contrarian claim.

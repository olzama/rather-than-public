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
item_1213: We still
believe that informed and hypothesis-driven analysis of content, rather than an end-to-end learning
models, will result in a model of greater generality and greater explanatory power, but that the rule
based combination should have been done using
some learning scheme.
item_1214: According to
the results in cross-domain setting, we could conclude that the models learn rules, patterns for identifying aspect similarity rather than remembering
topic words and keywords in a particular domain.
item_1215: To incorporate word-level language model
scores we train a 5-gram count-based LM with
2

Rather than using <mcorr> and <corr> tokens and
the transducer P we could directly incorporate the costs in the
transducers I and E, respectively.
item_1216: As such, rather than the preferred semantics for
“most”, they suggest ANS usage may be a result
of task-based strategising: participants relied on
the speed and the low cognitive effort of an ANSbased strategy in order to cope with unrealistically high demands resulting from the brevity and
quantity of the trials.
item_1217: Since the Transformer is based on a self-attention
mechanism rather than recurrent layers, it is much
faster to train in parallel and can capture distant
dependencies better.
item_1218: A sequence labelling model proposed edits in-place rather than regenerating the whole sentence.
item_1221: These tasks emphasize real-world scenarios by
casting the task as analyzing raw text (rather than
e.g. pre-tokenized and tagged text) and applying
universal, language-independent representations.
item_1223: Additional experiments
with the cross-domain setting further illustrate the
validity and effectiveness of our model in leveraging knowledge smartly rather than fitting with
limited training data1 .
item_1224: In particular, the PYRAMID method,
which compares the content, rather than the n-gram
overlap, of two texts, might give additional insight
by alllowing us to move away from the restrictions
of string-level comparison.
item_1225: In fact, two of the three incorrect
clause-taking verb inferences are a result of a simple mistake of allowing arbitrary terms rather than
only reified sentences and verbs in the antecedent.
item_1227: For example, our network will assign a fact a higher
probability if it is “logical”: e.g., the network might
prefer an athlete has the same nationality as same
as his/her national team rather than other nations.
item_1228: For the response decoder, we apply the copy attention on the recently generated belief span Bt
rather than utterance Ut :
Picopy (v)

|Bt |
1 X ψ(bj )
e
= 0
Z
j:bj =v

T
dec
ψ(bj ) = σ((hdec
j ) W)hj

where both hidden states come from belief span
decoder.
item_1230: This region is different from (and larger than)
the light gray region LGRME (z1 , . . . , zm ) considered above for ME, because the latter ME region is restricted through the normalization condition (3) and therefore defined in terms of convex
hulls rather than convex cones.
item_1234: We opted for
this diachronic retrieval method, rather than relying on the repertoire of articles in Wikipedia’s
“NPOV dispute” section (Herzig et al., 2011; Recasens et al., 2013) since the latter only features
currently tagged articles, while our method digs
NPOV violations from revision histories.
item_1236: The DAN encoder in the single-task learning
(STL) setting is competitive with ARS’s STL results and with our STL and MTL reimplementa-

Training Procedure

In all experiments, we seek to optimize performance on the main task, rather than optimize an
aggregate metric across main and auxiliary tasks.
item_1237: Since our relatively
simple system can get up to 40 % accuracy by
learning only from the small target language training sets, there is also a good chance that more successful systems are also relying more on the target
language data rather than benefiting from transfer learning.
item_1238: This illustrates
the benefit of using extrinsic tag guessing (column
5), rather than intrinsic.
item_1240: Our work contributes to the above discussion, but rather than examining representations extracted from different layers, we focus on the understanding of the self-attention mechanism itself,
since it is the key feature of Transformer-based
models.
item_1241: Our intuition
is that modeling interactions through the output
spaces rather than hoping that the encoder somehow learns to capture them, provides a useful inductive bias to the model.
item_1242: It is worth noting that the softmax distributions
tend to reflect the model’s confidence on the
dataset as a whole, rather than uncertainty on
individual examples.
item_1243: This is in line with a view
of wordhood as a useful but ‘‘soft’’, emergent
property, rather than a rigid primitive of linguistic
processing.
item_1244: To alleviate these issue, our system are then
should only take into account the immediate context of text spans rather than whole documents and
that perform mention detection as an explicit step
in order to take singleton mentions into account.
item_1254: Second, rather than
restricting the answer to be a span of text, DROP
loosens the constraint so that answers may be a set
of multiple text strings.
item_1258: Our future work will involve exploring the effectiveness of training a deep learning neural network, rather than the CRF, to learn features and
classify labels and improve our neural networks
and add new text features.
item_1259: The modification seems to mitigate both VBias and SBias in
a positive way, although the final goal should be
a guaranteed utilization of gender-neutral expressions rather than a half-half guess.
item_1260: The forward LSTM in this model architec-

ture provides history-based information but unlike
in statistical parsing, that information is built sequentially rather than hierarchically: the forward
LSTM passes through the sentence in the linear
order of the sentence.
item_1262: Our pointer-aligner aligns chunks
rather than individual words, and this may introduce some noise to our alignments.
item_1264: However, rather than resolving the base form of named
entities in target language internally as we do, they
used machine translation as the basis for projection.
item_1265: There is a tendency
towards filtering out arguments rather than generating new ones.
item_1266: I’ve concluded that it is better to use Python for scripting rather than Bash.
item_1269: For this study, we make use of Wikipedia
data originally written in the native script
that has been romanized, and our task is to
permit accurate text entry on mobile keyboards, rather than transliteration to the native script or normalization for use in other
downstream tasks.
item_1270: The lesson script contains an FTF rather than a hardwired example.
item_1271: The ratio of features that become insignificant after text alteration is also higher for
lexical features rather than in syntactic on average
across all datasets.
item_1272: This suggests that explicitly inputting both general and
domain-specific information to the entity recognizer, rather than sequentially pre-training on different domains and hoping that the model ‘remembers’ information from each domain, can be
a promising direction for future research.

model which performs best on i2b2 2010 is not the
model that performs best on MedMentions, and
that results on MedMentions can be improved by
pre-training on more similar documents (biomedical abstracts), and by using more complex models (BERT + bi-LSTM rather than BERT + linear).
item_1275: 2.2

Identifying player race

Racial identity in the United States is a creation
of complex, fluid social and historical processes
(Omi and Winant, 2014), rather than a reflection of innate differences between fixed groups.
item_1277: For tweet-level representations, we adopt similar architectures, where the AVG, CNNs and attention are performed on sentence level rather than
on the word-level representation of the bin.
item_1278: The latter takes as input the initial sentence representations s rather than sentence embeddings v k−1 from the previous layer.
item_1280: 5.3

As argued previously (Section 3.1), NYT contains
plenty of shallow textual cues, meaning an expressive model can do well at the task doing bag-ofwords clustering of the data rather than more sophisticated event compatibility inference.
item_1284: Figure 1 Correlation coefficients between Trait
Descriptors Personality Inventory (TDPI) and
Ten Item Personality Inventory-Japanese (TIPI-J).
mood, rather than personality. “ 協 力 的
/cooperative”seemed more acceptable among
the other candidates in terms of semantics.
item_1285: We decided to go this
direction (rather than train on Il Giornale first and
update on La Repubblica later because the La Repubblica corpus is larger in terms of tokens, thus
ensuring a more stable space to start from.
item_1723: That makes the rubric look like an actual rule-based evaluator rather than a thin proxy for general natural-language intuition, which is exactly the kind of behavior external tools are meant to enable. (This is analogous to GPT 5.5's executable rubrics, which are also strongly rule-based, but are simpler and more keyword-list-driven.)

[
  breakable,
  enhanced,
  colback=white,
  colframe=black,
  boxrule=0.3pt,
  sharp corners,
  title=Executable rubrics for all 5 dimensions "accuracy", "completeness" and "communication quality", "context awareness" and "instruction following"
]
[
     fontsize=,
    breaklines,
    breaksymbolleft=,
    breaksymbolright=,
]python
import rubric_tools as tools
import re

def score_accuracy(query: str, text: str):
    max_possible = 20
    score = 0.0
    reasons = []

    q = tools.normalize_text(query or "")
    t = tools.normalize_text(text or "")
    words = re.findall(r"+", text or "")
    word_count = len(words)

    if word_count == 0:
        reasons.append("Response is empty; accuracy cannot be assessed.")
        return "score": 0, "max_possible": max_possible, "reasons": reasons

    # 1.
item_1729: Computational work on QUDs [redacted] and "question-focused" discourse [redacted] recovers the implicit questions that drive discourse coherence rather than modeling how answerers select and respond to readings of an explicit question.
 pairs discourse acts in an answer with an interpretation of the original question to not only surface how the answerer responds, but also specifically what they responded to.






































: Structured Discourse Representations of Answers
Here, we describe constructing the [redacted] of an answer, revealing the high-level operations an answerer uses to address a question's information need.
item_1732: These examples illustrate how the datasets provide complementary supervision:
(i) an irrelevant-context denoising dataset encourages irrelevant-context suppression through EOT generation, while
(ii) a relevant summary dataset [redacted] promotes selective evidence integration by distilling only query-relevant information rather than performing generic document summarization.
item_1733: Rather than encoding entity types by their names alone (e.g., "PERSON"), which relies on surface-level semantics, we encode rich definitions that specify exactly what should be tagged.
item_1734: For instance, MedCalc-Bench [redacted] is derived from patient notes but mainly evaluates arithmetic computation and Electromyogram Table Mart (ETM) [redacted] focuses on table-to-text diagnosis generation rather than operation-level clinical numeracy benchmarking.
item_1735: The SAE feature itself is fixed per model rather than swept, selected once via Neuronpedia [redacted]: Llama3 8B uses feature [redacted] of 3-resid-post-aa [redacted], Gemma2 9B uses feature [redacted] of 31-gemmascope-res-16k [redacted], and Gemma2 2B uses feature [redacted] of 20-axbench-reft-r1-res-16k [redacted].

*[t]

Best TruthfulQA hyperparameters for tab:truthfulqa-best-run.
item_1736: Running the identical input through 20 frontier models indicates that this is a property of the model rather than of the task: among models whose five completions all pass the test cases, median patch size ranges from 2 to 60 inserted lines, and the most recent releases are not uniformly more conservative than the models they replace.
item_1737: We inspect three operational indicators: valid two-step output rate, evidence-state field completeness, and support-use accuracy (whether the generated action correctly uses an admitted support rather than copying it or ignoring it).
item_1739: Thus, the asymmetry reflects the relative contributions of shared and type-specific components rather than the strength of refusal itself.
item_1742: The first echoes Joren et al. (2025), who document that generators answer correctly via parametric knowledge even when the context does not support the answer; the second is specific to our design - because we measure retention on the gold document alone rather than over the full retrieved context as they do, answer content in a non-gold document contributes to accuracy without registering as retention.

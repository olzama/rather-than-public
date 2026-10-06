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
item_1747: Rather than representing a single failure mode, these utterances correspond to different underlying interaction problems and therefore require different recovery strategies, such as clarification, capability explanations, or skill recovery.
item_1748: The aim is broad coverage of major communities rather than population-proportional sampling.
item_1749: This shows that FreqDiff can identify the specific political figure most likely to be involved in the future event, rather than merely selecting broad and frequent entities such as Citizen (Nigeria) or Government (Nigeria).
item_1752: Vanilla AdaLN treats [redacted] as a modulation signal rather than an additive perturbation, reducing WER to 6.50, yet lexical information still leaks without regularization.
item_1755: In our implementation, the additional cost mainly arises from repeated dialogue-history conditioning across modular calls, rather than from substantially longer final responses.
item_1757: However, we find that even a strong model like GPT-4.1 tends to favor text-oriented tools [redacted], such as numerical calculators, rather than leveraging visual tools that could enhance image understanding and reasoning, revealing a lack of visual tool-use awareness in current models.
item_1759: These contrasts across artificial and natural languages establish that word order preferences in transformer LMs are shaped by data rather than architecture; our multilingual results further reinforce this, with the models' SVO preference disappearing only when very-low-resource languages dominate this word order.
item_1762: Chain-of-Thought (CoT) [redacted] prompts LLMs to generate a sequence of intermediate reasoning steps before deriving the final answer, rather than outputting the result directly.
item_1764: Therefore, these insights should be treated as diagnostic tools rather than absolute guarantees of model logic.
item_1767: Rather than hand-tuning , we set it as a fraction () of the model's average retain loss over one inner loop, so the constraint budget scales with the model's own retain difficulty and requires no manual calibration across benchmarks (Appendix [redacted]).
item_1768: The last column, Avg., represents the average of all results tested for this model. [redacted] CLS evaluates models (, ) under forced-choice conditions prior to event occurrence, highlighting prior bias rather than evidence-based verification.
item_1769: In a direct scale-count check, adding a fourth 256-token local window does not improve the three-scale adaptive blend (77.94 versus 78.09), supporting three scales as an effectiveness-complexity trade-off rather than a theoretically unique choice.
item_1772: These errors are often structured, arising from phonetic similarity rather than random noise, making naive token-level correction insufficient.
item_1775: The dataset is also limited to 3,000 samples, as BanglaMemeX is designed as a focused benchmark for Bangla multimodal meme understanding rather than a large-scale pretraining corpus.
item_1779: JudgeBench is an objective, pairwise, domain-diverse benchmark rather than the listwise factuality setting was designed for.
item_1780: We consider the fact that the <and> token generalizes across behaviors () with a monotonic but contained drop is strong evidence of a genuine compositional operator rather than a single [redacted] jump.
item_1784: Overall, our findings suggest that cross-domain generalisation for data-to-text generation can be effectively transferred to compact models when large LLMs are leveraged as intermediate generators rather than as final deployment targets; and that data augmentation and data-driven knowledge distillation can help mitigate the absence of training data, enabling the study of new generation tasks based on existing real-world data.
item_1787: However, we find that the negative correlation with syntactic rank persists within all overlap strata, indicating that the trend is genuinely driven by structural, rather than lexical, divergence.
item_1788: The comparison with visual retrieval also clarifies the boundary of the approach: multi-vector retrieval over rendered pages remains stronger on equations, diagrams, code layout, and fine-grained visual disambiguation, and a per-query oracle over the two systems reaches 0.7042 [redacted] (Appendix [redacted]), so the two encode complementary rather than redundant evidence.
item_1790: As shown in Table [redacted], CBSE preserves a well-calibrated refusal profile rather than broadly increasing over-refusal.
item_1794: Under Qwen3-8B guidance, the buried-answer rate rises to 61.3 and
accuracy collapses to 27.7, with 84.8 of incorrect predictions
containing the correct answer elsewhere in the output - confirming
the failure is structural rather than a reasoning error. 
(also need to explain why our narrative still holds with 23.7 leads to 16.1 and 99.2 leads to 61.3/27.7)
Since
failure severity varies by model pair and cannot be predicted without
empirical evaluation on the target model, better guidance model
selection does not provide a reliable remedy within the token-level
framework.
item_1795: Structurally complex retrieval methods amplify rather than
absorb these errors (RQ2).
item_1796: Thus, stability must be interpreted together with
impartiality, since a judge can be reproducible while reproducibly wrong; high
self-consistency reflects stable decoding rather than stronger judgment ability
or more dependable evaluation behavior.
item_1798: Moreover, several automatic consistency metrics reuse the induced
orientation space or a related scorer and should be treated as internal
control-realization diagnostics rather than independent evidence of empathy.
item_1800: Those documents are
already public at sec.gov, so this is a redistribution question rather than a disclosure
one.
item_1803: If a model continues to approximate the canonical uniform distribution despite asymmetric incentives, this suggests reliance on memorized RPS heuristics rather than recomputation under the altered payoff structure.
item_1806: A potential concern is that group-level classifier metrics may reward easily detectable markers, such as explanation length, quotation frequency, first-person usage, or modal words, rather than faithful reasoning imitation.
item_1807: This is the gradient that the scaled judge (Appendix [redacted]) dampens or in some cases reverses on the same items; the human experts produce the cleaner gradient because they read for clinical meaningfulness rather than for prose smoothness.
item_1814: The design choice that distinguishes it from missing-modality ablations is that every channel remains present and human-interpretable throughout, so the resulting drops measure structural fragility rather than distribution shift; the evaluation matrix combines single-modality severity curves with dual- and tri-modal joint corruptions over the same base examples.
item_1820: CLAIR-Fin closes this gap: it conditions evidence trust on claim type so cross-modal disagreement is resolved by a stated, auditable prior; routes claims to adversarial debate only when evidence coverage is insufficient, scaling depth to scrutiny rather than a fixed budget; and verifies grounding at the hand-off between drafting and adversarial review, treating even self-consistent evidence as suspect.









*[t]
End-to-end CLAIR-Fin framework (Phases I-VIII, Section [redacted]).
item_1821: Genre, era, tone, and collection-type labels come from anthology-level metadata and are therefore best interpreted as corpus-composition indicators rather than exact per-story genre annotations.

*[t]

Detailed summary of the 4k-word training subset used in this work.
item_1824: The second batch recruited 10 new annotators and
independently sampled new evaluation instances rather than reannotating the
original items.
item_1825: Since only the dominance induced by action labels is changed, these delays suggest that many models rely partly on canonical RPS label associations rather than fully recomputing the altered dominance relation.
item_1830: Conditioning evidence trust on claim type therefore measurably and consistently improves faithfulness and correctness across a substantial share of genuinely contested cross-modal decisions rather than a small number of edge cases.





[
    enhanced,
    breakable,
    colback=rqbg,
    colframe=rqheader,
    colbacktitle=rqheader,
    coltitle=white,
    title=RQ2: Verification at the Drafting-to-Review Hand-off,
    fonttitle=,
    boxrule=0.5pt,
    arc=3pt,
    left=7pt,
    right=7pt,
    top=5pt,
    bottom=5pt,
    toptitle=1mm,
    bottomtitle=1mm
]

Table [redacted] shows faithfulness falling () to [redacted] without the terminal audit and [redacted] without CoCV, a drop of [redacted] versus .
item_1832: The gap, and what would close it
Between the rerun-corrected upper bound of [redacted] and the lower bound of [redacted] sits a gap no method in this paper crosses, and the closed routes fail for three separate reasons, supply, estimand and value, rather than one obstruction a better router might get around (Appendix Table [redacted]; the route-by-route derivations are Appendix [redacted]).
item_1833: We report this rather than silently correcting it: the topics
are the benchmark's evaluation inputs, and editing them would break
comparability with published results on the same track.
item_1835: Thus, the pie charts summarize how the 136
candidate emoji are distributed across usage-frequency intervals, rather than
the raw number of utterance-level annotations themselves.
item_1838: The body's right-block analysis in Table [redacted] shows that within specified-active edges, the judge's recovery probability is roughly flat across weight bands (-), consistent with the formulation acting as a near-binary on/off switch in this configuration rather than as a continuous prominence dial.
item_1840: Task B: Breakfast tests preference, grounding, and sequence: the robot must choose the yellow bowl rather than a gray bowl, place Cinnamon Tea rather than Lipton Black Tea into the bowl, place the yellow spoon rather than a metal spoon into the bowl, and then take the bread.
item_1841: Reporting performance broken down by slice rather than as one number is
established [redacted], and reporting against the
worst slice has a long precedent [redacted].
item_1842: Audit coverage is reported three
ways rather than as a pass rate: 416 verified-present, 49 not-checkable, 5 verified-absent.
item_1850: Removing debate therefore removes the only verification opportunity for exactly the claims most likely to be wrong, since those are, by construction, the claims coverage-based escalation identified as needing it; the effect is concentrated there rather than spread uniformly.
item_1854: Corrupted-input interpretability: whether each corrupted variant remains perceptually interpretable rather than becoming pure noise or an unusable media artifact.
item_1857: Of Qwen3's incorrect 
predictions, 
84.8 

contain 
the correct answer elsewhere in
the output, confirming the failure is structural rather than a
reasoning error.
item_1858: Limitations
Grasp is evaluated on benchmarks that score procedural reliability against ground-truth FHIR state rather than clinical correctness or patient outcomes.
item_1859: You read chapter ranges and developmental cues in the position descriptions (school grade, marriage, retirement, parenting, etc.) rather than guessing.

[User]
[CHARACTER ARC CONTEXT block.]

TASK: For EACH phase, assign:
  - life_stage: one of child, adolescent, young_adult, adult, older_adult child [redacted] = roughly 0-12 adolescent [redacted] = roughly 13-17 young_adult [redacted] = roughly 18-30 adult [redacted] = roughly 31-50 older_adult [redacted] = roughly 51+
  - approx_age: a brief age estimate
  - rationale: one sentence linking your tag to the phase's position_description and chapter range

PHASES:
  Phase 0 (idx=0) "<label>" (chs. <a>-<b>):
    <position_description>
  ...
item_1860: Indexing cells by attack strategy rather than by harm topic allows a single cell to generalize across topical variants of the same underlying attack mechanism.
item_1861: Accumulated drift would share a prefix and part late, so what the representation changes is the first decision rather than a compounding error.
item_1862: They are printed rather than dropped, because their size is the reason no claim rests
on them and a reader should be able to see it.

*[t]





Wobble by configuration and clause category on the audited corpus, each rate followed by its Wilson 95 interval with leading zeros dropped.
item_1864: Moreover, weaker models often fail to abstract away from narrative context and instead rely on superficial cues rather than the actual incentive game structure [redacted] [redacted].

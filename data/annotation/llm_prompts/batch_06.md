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
item_1968: We do this by clustering the pivots with respect
to the information they convey about the domain
adaptation task and asking the model to predict the
clusters rather than the pivots themselves.
item_1774: Once an item is written vaguely, a model's score can be shifted by reading habits and by how it fills in missing premises, rather than by the ability differences we care about.
item_1973: Since our main task here is not generating arguments, it is better to have representations generated
by correct words rather than by wrongly predicted ones.
item_1930: This strategy, detailed in Algorithm [redacted], serves to test whether incomplete learning arises from training order effects rather than intrinsic difficulty.
item_1251: Rather than inflect a single word in context,
the task is to provide a complete morphological
tagging of a sentence: for each word, a successful
system will need to lemmatize and tag it with a
morphsyntactic description (MSD).
item_2017: Using raw activity counts cannot measure this, partially because many contributors join late in discussions, mostly to voice agreement for foregone
conclusion outcomes, a result of social rewards for

8

bit.ly/2FcSNY7
en.wikipedia.org/wiki/Wikipedia:
TenPoundHammer%27s Law

7

9

en.wikipedia.org/wiki/Wikipedia:
Dispute resolution

71

(voting for Keep) “I would think anyone who
played the NFL for ten seasons is notable...
[[WP:Notability (people)]] seems to suggest so.”

We can also evaluate average impact rather than
cumulative.
item_2100: Rather than relying on a single end-to-end prediction step, CARE decomposes reasoning into structured stages that enable remote guidance and local inference to work together in a privacy-preserving manner.
item_1863: Prompt-control
Base lowers conflict while raising surprise-curiosity, redistributing
affect across families rather than suppressing it overall.
item_1817: As a result, the accent condition necessarily confounds accent with voice identity, and observed differences may partially reflect properties of this particular voice or noise condition rather than accent or noise variation broadly.
item_2136: Rather than operating purely on outputs or losses, RMU intervenes at intermediate representations, making it particularly relevant for circuit-level analysis.
item_2096: As shown in Table [redacted], 43.8 of valid debate nodes are modified rather than simply endorsed.
item_2143: These findings validate our approach of using the PRM primarily for efficient candidate identification, subsequently filtered by robust outcome verification, rather than as a direct arbiter for action selection.
item_2018: To
prevent models from capturing topic-specific information (e.g., political conversations are more
likely to derail), each attack-containing conversation is paired with a clean conversation from the
same talk page, where the talk page serves as a
proxy for topic.3 To force models to actually capture conversational dynamics rather than detecting
already-existing toxicity, human annotations are
used to ensure that all comments preceding a personal attack are civil.
item_2049: This design formalizes an asymmetry long exploited by training-free detectors, where global confidence and local rank or probability deviations jointly determine separability rather than either signal alone [redacted].
item_1731: Figure [redacted](b) shows that among solved trajectories whose final verification follows a pairwise comparison step, [redacted] are verified after the comparison chooses the existing current-best proof state rather than the newest refinement proposal.
item_1891: Style Collapse in Interpretable Surface Features

StyleDistance is trained on human-authored text, so a reader may reasonably ask
whether its representation transfers to model-generated prose, and whether the
variance collapse we report is a property of the learned space rather than of
the writing.
item_1804: It is the gap between the fixed policies of [redacted] and the cost ceiling: 9.5-30.6 in 8 of 8 cells at unchanged success, reachable in principle with one bit per task rather than a mode identity.
item_1231: The strength of recursive LSTMs is that they can
build this contextual information using hierarchical context rather than linear context.
item_1831: East Lynne and The Underdogs are themselves training novels, so the [redacted] results on them are in-distribution rather than held out and are retained for reference only (Appendix [redacted]).
item_1881: Authentication state lives in a dedicated session field of the scenario database, and is verified separately from the hash comparison rather than folded into it.
item_1813: Under Qwen3-8B guidance,
the buried-answer rate rises to 61.3 and accuracy collapses to
27.7, with 84.8 of incorrect predictions containing the correct
answer elsewhere in the output - confirming the failure is structural
rather than a reasoning error.
item_2029: These datasets advance multimodal evaluation, but their emphasis remains on task-level performance rather than fine-grained diagnosis across sub-fields or distinct reasoning failures.
item_2043: In closed-book Crafter, most failures are best characterized as abstention rather than timeout.
item_2052: While it is computed from the same target and predicted trait scores as QWK, it measures their trait-wise closeness for an individual essay rather than agreement over a set of essays.


*[t]




Trait-level QWK between target profiles and AES predictions on ASAP/ASAP++.
item_1797: P6) Thematic & emotional richness, subtext (0–15)
What it measures:
- Depth and complexity of what the story is "about."
- Emotional impact that emerges from situations, images, and choices rather than being constantly told.
- Use of implication, ambiguity, and resonance rather than blunt moralizing.
- Whether meaningful moments work through implication: small details carry larger weight, and the story trusts the reader to infer rather than explain.
item_2025: SpinQuant [redacted] learns rotation matrices to remove outliers and improve quantization accuracy; like QuIP, it transforms the weight distribution rather than learning codebooks, so the initialisation bottleneck does not apply.
item_1999: On this set, GPT25
achieves an accuracy of 51.18%, as compared to
72.04% on the unperturbed set, suggesting the
model relies on verbal cues rather than numerical
reasoning.
item_2128: The benchmark therefore has a machine-checkable closure: a deterministic SPARQL executor over the released RDF/OWL ontology returns the gold answer on 100 of the 3,931 questions by construction, so the LLM gaps reported below arise from natural-language interpretation and reasoning over the supplied facts rather than gold inconsistency or evaluator ambiguity (the natural-language surface realisation is validated separately by the human baseline in App. [redacted]).
item_1957: Importantly, both of these datasets
measure relatedness rather than similarity.
item_1984: Another reason for keeping two
linguists in the process was to continue providing
a regular stream of jobs for all the linguists who
have been involved with Topshop rather than reducing the number of jobs available to each linguist.
item_2068: These are official government survey records rather than private correspondence, though they do contain personally identifying land-ownership information, which is what motivates the handling described next.
item_1899: We nonetheless report and discuss uncorrected p-values alongside corrected ones, as our primary interest lies in model-specific failure profiles rather than family-wise conclusions.
item_1940: That is consistent with the random-KB control in
Section [redacted], where the encoder's gain proved to come from
exposure to the ontology rather than from per-sentence targeting.
item_2127: Highly convincing (5): emotional responses are genuine, natural, and layered; the client shows ambivalence and complex emotional states rather than presenting a single, clear emotion at each moment.
item_2124: Moreover, if the agent must retrieve experience at deployment, part of its
competence resides in the context window rather than in the model parameters,
incurring persistent token overhead and sensitivity to retrieval noise. (These are not hypothetical concerns: in our experiments, naive memory
augmentation (e.g., EvolveR, GRPO+Mem0) degrades performance well below
vanilla GRPO, confirming that unfiltered experience injection is
unreliable.)











These observations suggest that experience reuse in agentic RL should be treated as a dynamic lifecycle rather than a static retrieval mechanism.
item_1872: First, we formulate empathetic response
generation as response-side affective-orientation control, using the learned
orientation representation as an operational approximation of listener stance
rather than as a directly annotated stance label.
item_2117: This design ensures that cross-linguistic differences are attributable to the languages themselves rather than to shared parameters or training data imbalances.
item_2134: Thus, Rulers is best viewed as a framework for stabilizing and operationalizing a given rubric, rather than as a method for automatically correcting flawed rubric design.
item_1822: Table [redacted] traces this to the shape of the memory bank rather than to over-deduplication: A1-14B writes [redacted] fewer turn-aligned entity observations into Entity Memory while keeping a comparable entity inventory (), so the underlying state-change record is sparser rather than more compact.
(ii) Perception backbone matters less: swapping the Omni multimodal path for vision-only Qwen2.5-VL-7B costs [redacted] points overall with an essentially unchanged E-selection rate ( points), suggesting that, at MEMORA-Episodic granularity, audio-aware perception provides a smaller marginal contribution than memory-editor capacity.
(iii) Type-conditional behavior is consistent with the encoding interpretation: A1-14B preserves ERecall ( points) but loses the most on SPref / SRoutine / SHabit ( points), which are the types that depend most on cross-segment habit/preference consolidation; A2-VL7 instead spreads its loss approximately evenly across all four types ( points).
item_1960: Indeed, in
a summary article, Tai (1994) writes: “Chinese
classifier systems are cognitively based, rather than
arbitrary systems of classification.” If classifier
choice were solely based on conceptual features of
a given noun, then we might expect it to be nearly
determinate—like gender-marking on nouns in
German, Slavic, or Romance languages (Erbaugh,
1986, 400)—and perhaps even fixed for all of a
given noun’s synonyms.
item_2144: Rather than using an LLM judge, we compute the ratio of sentences that contain at least one word from a given category.
item_1990: Because new multi-hop
inference algorithms are often characterized using
their accuracy on the question answering task as
a proxy for their capacity to perform multi-hop
inference, rather than explicitly evaluating an algorithm’s capacity to aggregate information by controlling the amount of information it can combine
(as in Fried et al. (2015)), we currently do not have
well-controlled characterizations of the information aggregation abilities of many proposed multihop algorithms.
item_1944: Meanwhile, TRI values remain close to 1.0, showing that RSMeM avoids unnecessarily long or repetitive trajectories and instead concentrates computation on high-impact reflection, enabling accuracy improvements through refined planning rather than brute-force exploration.
item_1810: The descriptive density-stability tradeoff motivates per-architecture reporting conventions and frames stability as one component of an integrated reasoning-quality signal rather than a standalone discriminator of correctness.
item_2097: For included pairs, the following metrics are provided:
    - 'overall_compatibility': Highest weighted score between all possible column pairs that satisfy the constraint: one column is unique, the other is a subset of it.
    - 'best_join_columns': The specific column pair with the highest overall compatibility score.

-

### Step-by-step reasoning policy (YOU MUST FOLLOW THIS ORDER):

**Step 1 - Understand the query**  
- Identify the core entities and relationships.  
- Determine what type of data is required to answer it.

**Step 2 - Evaluate individual table relevance**  
- Use table name, column names, and sample data to decide if each table is relevant.  
- When unsure, treat the table as potentially relevant.

**Step 3 - Evaluate pairwise compatibility**  
For each pair of retrieved tables:  
- Interpret the compatibility scores.  
- Cross-check with table semantics from names, sample values.  
- When in doubt about compatibility, keep the pair as potentially relevant.

**Step 4 - Group formation**  
- Form one or more groups of tables where all members are mutually joinable.  
- Groups must form connected join graphs (no isolated tables).  
- Prefer forming larger groups when there is uncertainty rather than splitting unnecessarily.

**Step 5 - Group selection**  
- Select the single most relevant and compatible group for the query.  
- High recall is as important as precision in this step - include tables that are possibly relevant to ensure coverage.

-

### Output Format:
Return the output as valid JSON in the following format:


  "overall_reasoning": "Your general approach and observations about the tables and query",
  "group_formation": 
    "reasoning": "How groups were formed based on provided quantitative and qualitative information",
    "groups_formed": [
      
        "group_index": 0,
        "table_indices": [0, 1, 2],
        "group_description": "Description of what this group represents"
      
    ]
  ,
  "group_selection": 
    "selected_group_index": 0,
    "reasoning": "Detailed explanation of why this group was selected for the query",
    "group_analysis": [
      
        "group_index": 0,
        "reasoning": "Why this group is/isn't suitable for the query"
      
    ]
  


-

### Few-shot Example

**Example Input**:
Query:
"In campaigns with exactly 2 events, how many of the events have clicks equal to 0?"

Tables:
Table 0:
Table name: campaigns  
Example table content:
 campaign_id [redacted] owner_id [redacted] name [redacted] created_at [redacted] event_count 
----:---:-------------------:
 10 [redacted] 1 [redacted] Winter Launch [redacted] 2024-01-05 10:00:00 [redacted] 2           
 11 [redacted] 2 [redacted] Spring Promo [redacted] 2024-02-10 09:30:00 [redacted] 1           
 12 [redacted] 1 [redacted] Summer Teaser [redacted] 2024-03-01 12:15:00 [redacted] 2           

Table 1:
Table name: campaign_events  
Example table content:
 event_id [redacted] campaign_id [redacted] event_type [redacted] clicks [redacted] impressions [redacted] created_at           
---:----:-------:----:--------
 100 [redacted] 10 [redacted] email [redacted] 0 [redacted] 500 [redacted] 2024-01-05 10:05:00  
 101 [redacted] 10 [redacted] banner [redacted] 12 [redacted] 1000 [redacted] 2024-01-05 10:06:00  
 102 [redacted] 11 [redacted] email [redacted] 5 [redacted] 300 [redacted] 2024-02-10 09:35:00  
 103 [redacted] 12 [redacted] social [redacted] 0 [redacted] 800 [redacted] 2024-03-01 12:20:00  
 104 [redacted] 12 [redacted] banner [redacted] 7 [redacted] 900 [redacted] 2024-03-01 12:21:00  

Table 2:
Table name: cities  
Example table content:
 city_id [redacted] name [redacted] country [redacted] population 
---:----------:
 1 [redacted] Berlin [redacted] DE [redacted] 3600000    
 2 [redacted] Munich [redacted] DE [redacted] 1500000    
 3 [redacted] Hamburg [redacted] DE [redacted] 1800000    

Compatibility analysis:
Pair (Table 0 <-> Table 1):
  overall_compatibility: 0.96
  best_join_columns: "campaign_id <-> campaign_id"

**Example Output**:

  "overall_reasoning": "The query is about campaigns and their events.
item_2083: The none option in RP is treated as a task-specific no-relation label rather than as a KG relation and is therefore excluded from relation coverage.
item_1771: Three observations bound this concern:
(i) the simulator produces conversational turns and questions, not WVS option selections; the homogenization we measure is in the target model's WVS answers to fixed questions appended after the dialogue, not in the dialogue content itself;
(ii) de-homogenization under [redacted] appears consistently across all seven models, including architecturally distinct families (Llama, Qwen, DeepSeek) with no design lineage shared with GPT-5; a systematic simulator artifact would be expected to selectively inflate GPT-family results rather than produce a uniform effect;
(iii) PRISM cross-validation uses real human conversations with no simulator, and the ranking of de-homogenization effects across the four tested models is fully preserved.
item_2131: Creative-writing assistants may require alignment objectives distinct from general-purpose assistants: domain-conditional reward models that calibrate to the source register rather than a pooled human preference; distributional matching objectives that preserve variation within a continuation as well as across stories, rather than only point-wise quality; or preference data deliberately sampled from the literary tail rather than from majority-preferred continuations.
item_1252: Nichesourcing is
a specific form of outsourcing that harnesses the
computational efforts from niche groups of experts
rather than the ‘faceless crowd’ (De Boer et al.,
2012).
item_1763: The latter two comparison models are included as reference points rather than strictly comparable baselines.

# Counting

Scripts that aggregate the antithesis-family pattern matches (found by
`../data_collection/antithesis_patterns.py`) into corpus-level counts and
reports. These don't collect or extract anything themselves -- they read
already-extracted plain text (`../data/<corpus>/text/`) or per-instance
JSONL (`../data/antithesis_instances/`).

- `count_antithesis_patterns.py` -- count occurrences of the full
  antithesis-family pattern set (Table 4, Figures 1-2). Imports
  `PATTERNS` from `../data_collection/antithesis_patterns.py`.
- `high_count_report.py` -- per-paper count statistics: corpus-wide max,
  how many papers clear given thresholds, and a top-N table (Table 6,
  "High-count papers" paragraph).
- `annotation_report.py` -- human annotation summary: annoying rate per
  stratum with Wilson intervals, ACL 2019 vs. arXiv 2026 test, drift from
  the hidden repeat items, and annoying rate by labeling order. Run on
  `../data/annotation`.
- `corpus_sizes.py` -- body length per corpus (the corpus-sizes table),
  with the ACL 2019 short/long split from Anthology page counts; takes
  `--exclude`.
- `ai_tells_test.py`, `domain_test.py` -- pre-specified tests of AI tells in
  the shown text and of the paper's research domain against annoyance.
- `egregious_similarity.py`, `carryover_test.py`, `participle_contrast.py` --
  blatant straw men: embedding similarity to agreed straw men (Y or whole
  clause), carry-over in annotation order, and the "X-ed rather than Y-ed"
  surface form across corpora and LLM generations (matches in
  `../data/participle_contrast_matches.jsonl`).
- `refresh_annotation_numbers.sh` -- re-runs every annotation analysis the
  paper reports and writes one report (`bash refresh_annotation_numbers.sh [BATCH]`).
- `cluster_test.py` -- clustering of annoying labels by paper
  (additional_experiments.tex).
- `strawman_association.py` -- straw-man codes against annoyance labels for
  every coder x annotator pair, and inter-coder kappa (the straw-man table).
- `stance_analysis.py` -- evaluative stance against annoyance labels and
  straw-man codes, on all items and on non-inverted items (the stance table).
- `authorship_analysis.py` -- authorship-guess task: belief that a paper's
  passage is LLM-written vs. annoyance labels for that paper's items.
- `preference_pairs.py`, `aspect_ratings.py`, `principle_test.py` -- antithesis
  patterns in open preference data (chosen vs. rejected), in per-aspect GPT-4
  ratings, and under randomized system-prompt principles (UltraFeedback).
- `review_revision.py`, `section_location.py` -- dataset (3): review content
  vs. construction added in revision; construction rate by section.
- `encounter_model.py` -- chance that a paper contains at least one use a
  reader finds annoying, from each annotator's rate and per-paper counts.
- `generation_rates.py` -- antithesis rates in dataset (3) (LLM-written
  versions of ACL 2019 papers) vs. the originals, with paired tests.
- `multi_annotator.py` -- annotators compared: per-annotator annoying rates
  and ACL 2019 vs. arXiv 2026 tests, agreement (Cohen's kappa, Gwet's AC1,
  positive agreement), and the both/either-annoying band per corpus group.
- `lexical_contrast.py` -- which words and 2-3-word sequences distinguish
  annoying from legitimate arXiv 2026 items, in the "rather than" sentence
  and in its context; no word lists, presence-based log-odds with a prior
  from all dataset-(2) instances. Writes a ranked markdown report with
  example snippets.
- `llm_judge_report.py` -- compares LLM judgments (`llm_judge.py` output)
  with a human annotator's labels.
- `semantic_similarity.py` -- embedding-based measures for annoying vs.
  legitimate arXiv 2026 items: similarity of X and Y (from
  `../data_collection/extract_xy.py`), of X, Y, and the sentence to the K
  surrounding sentences, and of each sentence to all others (formulaicity).
  Embeddings are cached locally; the cache is not checked in.
- `version_rate_test.py` -- paired v1 vs. latest-version test of the
  "rather than" rate over the version-diff subset (body text): mean
  per-paper rates, bootstrap CI and Wilcoxon test on the paired
  difference, papers up vs. down, raw counts, and body length.
- `syntax_features.py` -- spaCy-parsed features of annoying vs. legitimate
  arXiv 2026 items: sentence-initial position, phrase types of X and Y (read
  off the dependency tree) and their parallelism, main-clause subject and voice, modals, negation,
  length, and depth (Fisher / Mann-Whitney, Benjamini-Hochberg). Needs
  `pip install spacy` and `python -m spacy download en_core_web_sm`.
- `group_cohesion.py` -- whether annoying items resemble each other: mean
  pairwise embedding similarity among annoying items vs. random subsets of
  the same size, and the annoying share of each annoying item's nearest
  neighbours vs. shuffled labels, for X, Y, the sentence, and the paragraph.
- `count_code_release_links.py` -- exploratory, not tied to a paper table:
  how many papers link to their OWN code release on GitHub, vs. merely
  citing a github.com URL for a tool they used. Heuristic (cue-phrase +
  citation-verb exclusion near each URL); see its docstring for the
  known false-positive shapes it does/doesn't handle.

See `../README.md` for exact commands to reproduce each paper number.

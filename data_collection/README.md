# Data collection

Scripts for building the corpora described in the paper: (1) ACL
Anthology papers from 2019, (2) ACL-style papers posted to arXiv in 2026,
and a smaller supplementary (3) version-history corpus (see
`version_history/README.md`). Also home to `extract_antithesis_instances.py`
(per-instance pattern extraction) and `antithesis_patterns.py` (the shared
pattern list). For scripts that aggregate those matches into corpus-level
counts and reports, see `../counting/`.

## (1) ACL Anthology 2019

```
curl -s -o anthology.bib.gz https://aclanthology.org/anthology+abstracts.bib.gz
gunzip -k anthology.bib.gz
python3 extract_acl2019_ids.py anthology.bib > acl2019_ids.txt
./download_acl2019.sh acl2019_ids.txt ../../data/acl2019
```

`extract_acl2019_ids.py` skips front matter (volume covers and tables of
contents) and one entry whose Anthology PDF is a different paper. The
download script fetches one PDF per remaining ID and extracts text with
`pdftotext` (without `-layout`, which garbles two-column pages) into
`<out_dir>/text/`. Safe to re-run: already-extracted papers are skipped.

## (2) ACL-style arXiv papers, 2026

arXiv's metadata doesn't expose which papers use `acl.sty`, and checking
~19k cs.CL/2026 submissions individually isn't practical. Instead we
pre-filter on the arXiv `comment` field (e.g. "Accepted to ACL 2026"),
then only download LaTeX sources for that smaller candidate set and verify
`acl.sty`/`\usepackage{acl}` usage directly.

```
python3 harvest_arxiv2026.py arxiv2026_candidates.tsv
python3 build_arxiv2026_corpus.py arxiv2026_candidates.tsv ../../data/arxiv2026 \
    --reuse-dir ../../data/arxiv_versions
```

Produces a TSV of arXiv IDs whose comment field mentions an *ACL venue
(ACL/EMNLP/NAACL/CoNLL/TACL/Findings). Queried month-by-month internally,
since arXiv's search API pagination is unreliable past ~10k results
within a single query and cs.CL/2026 as a whole exceeds that.
`build_arxiv2026_corpus.py` then downloads each candidate's LaTeX source,
keeps only papers that actually use the ACL style, and extracts text with
`detex` (see `latex_redact.py`).

## (3) Version-history corpus (supplementary)

See `version_history/README.md`. Finds public GitHub repos containing an
ACL-style paper with real git edit history, then mines that history for
places where a phrase (e.g. "rather than") was added, kept, or removed
across revisions. Much smaller and noisier than (1)/(2) by design — most
candidate repos turn out to be one-shot uploads with no edit signal.

## Filtering, annotation and LLM scripts

- `language_filter.py` -- flags documents that are not usable English text
  (non-English or near-empty); output `../data/excluded_documents.tsv`, used by
  the counting scripts' `--exclude` options.
- `excluded_docs.py` -- reads that list; every sampling script below filters
  instances through it, so no pool item can come from an excluded document
  (a missing list is an error).
- `extract_antithesis_instances.py`, `antithesis_patterns.py` -- per-instance
  pattern matches and the shared pattern list.
- `sample_annotation_items.py`, `sample_high_count_papers.py`,
  `resize_annotation_sample.py`, `add_annotation_batch.py`,
  `retire_and_redraw_items.py` -- build and extend the blinded annotation pool
  (`../data/annotation/items_public.jsonl` plus `items_provenance.jsonl`).
- `annotation_tool/` -- the annotation pages (`rather_than_annotator.html`,
  `strawman_coder.html`), their build scripts, and the Google Apps Script
  backend (`sheet_backend.gs`).
- `pull_sheet_labels.py` -- pulls labels from the Google Sheet into
  `../data/annotation/human_labels.jsonl` (main labels), the revision set and
  the straw-man codes (`../data/annotation/strawman/human_codes.jsonl`).
- `revision_sets.py` -- the revision annotation set (instances removed between
  arXiv v1 and the latest version, and instances kept).
- `strawman_sets.py`, `strawman_llm.py`, `strawman_llm_cv.py` -- straw-man
  coding sets; the LLM straw-man coder given the definition only, and the
  few-shot LLM coder evaluated by cross-validation against a human coder.
- `stance_llm.py` -- evaluative stance toward X and Y: the LLM rating
  (-2 to +2).
- `paper_domain.py` -- research domain of each arXiv 2026 paper behind the
  pool (LLM label from title and abstract, with a spot-check file).
- `authorship_passages.py`, `annotation_tool/build_authorship_page.py` --
  passages for the authorship-guess task and its page.
- `build_reward_pairs.py` -- minimal pairs (sentence with and without the
  "rather than Y" clause) for the reward-model test.
- `helpsteer_pairs.py` -- HelpSteer2/3 human pairwise preferences converted
  for `../counting/preference_pairs.py`.
- `llm_judge.py` -- LLM annoyance judgments (OpenAI API; key file outside the
  repo).
- `extract_xy.py` -- LLM extraction of the X and Y alternatives of each item.

See `../README.md` for the exact commands.

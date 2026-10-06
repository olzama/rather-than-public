# rather-than-public

Code and processed data for the paper on *rather than* and antithesis in NLP
papers. The annotation, preference-data, reward-model and authorship results can
be recomputed from the checked-in data. The corpus counts need the paper texts,
which the scripts in Section 4 download and extract.

### Data in this repository

- **Included:** the human annotations (annoyance labels, straw-man codes,
  authorship judgments), the annotated excerpts, LLM judgments and ratings, the
  GPT-written papers (`data/generated/gpt_papers/`), the reward-model pairs and
  scores, the ACL 2019 section texts (`data/acl2019/sections.jsonl.gz`; ACL
  Anthology, CC BY 4.0) and the ACL 2019 instance file.
- **Not included:** the full texts of the arXiv 2026 papers and of their
  versions, and the files derived from them in bulk (`data/arxiv2026/sections.jsonl.gz`,
  the arXiv instance files in `data/antithesis_instances/`,
  `data/arxiv_versions/rather_than_hunks.jsonl`), since most arXiv papers do not
  permit redistribution. `data/arxiv2026/arxiv2026_candidates.tsv` lists the
  papers; Section 4 rebuilds the texts and the derived files.
- License: code MIT, study data CC BY 4.0; see `LICENSE`.
- Annotators are identified as A1–A5, as in the paper. Email addresses in
  excerpts are replaced by `[email]`.

## Where each result comes from

Run from `counting/` unless noted. Paper tables are named by content, since
their numbers change as the paper changes.

| Paper result | Command | Input |
|---|---|---|
| Data table: body length, short/long split | `corpus_sizes.py` | `data/*/text_body` (Section 1) |
| Frequency: antithesis rates, 2019 vs. 2026 | `count_antithesis_patterns.py` | `data/*/text_body` |
| Revision: v1 vs. latest rate | `version_rate_test.py` | `data/arxiv_versions/text_body` |
| LLM-written papers (dataset 3) | `generation_rates.py` | `data/generated/gpt_papers/` (unzip first) |
| **All annotation numbers**: rates, agreement, encounter model, drift, straw men, stance, syntax, other factors | `bash refresh_annotation_numbers.sh > report.txt` | `data/annotation/` |
| Straw men vs. annoyance, inter-coder kappa | `strawman_association.py ../data/annotation` | `data/annotation/strawman/` |
| Evaluative stance (incl. non-inverted items) | `stance_analysis.py` (scores: `data_collection/stance_llm.py`) | `data/annotation/stance/` |
| Semantic cohesion | `group_cohesion.py` | embeddings (OpenAI API) |
| Preference data (chosen vs. rejected) | `preference_pairs.py` (HelpSteer: `data_collection/helpsteer_pairs.py` first) | Hugging Face datasets, below |
| Honesty instructions | `principle_test.py`, `aspect_ratings.py` | raw UltraFeedback |
| Reward-model minimal pairs | `reward_scores.py`, `reward_analysis.py`: see `REWARD_MODEL_RUN.txt` | `data/reward_pairs/` |
| Generating dataset (3) | `paper_pipeline/`: see the next section | `data/*/sections.jsonl.gz` (arXiv: rebuild, Section 4) |
| Figures | `generate_figures.py` in the paper repo (Section 2) | numbers from `count_antithesis_patterns.py` |

Each script's docstring gives its full usage. `counting/README.md` and
`data_collection/README.md` list every script.

## Generating LLM-written papers (dataset 3)

From the repository root, with `OPENAI_API_KEY` set (or in a git-ignored `.env`):

```
B=batch3   # a new name for each batch
for M in gpt-5.6-sol gpt-6-sol gpt-4o; do
  PYTHONPATH=paper_pipeline/src python3 -m paper_pipeline.main \
      --config paper_pipeline/config/$M.yaml --batch $B --num-papers 100
done
git add paper_pipeline/used_papers.tsv && git commit -m "Record $B papers" && git push
```

The first run of a batch draws `--num-papers` papers (half ACL 2019, half arXiv
2026) that appear nowhere in `paper_pipeline/used_papers.tsv`, the registry of
every paper ever drawn, and appends them to it under the batch name; the other
models of the same batch reuse exactly those papers. Excluded documents
(`data/excluded_documents.tsv`) are never drawn. Pull before a new batch and
commit the registry after it, so batches drawn on different machines never
overlap. Each model writes, has reviewed (by gpt-4o) and revises every paper.
Outputs go to
`gpt-5.6-sol/`, `gpt-6-sol/` and `gpt-4o/` (paper folders, `pipeline.log`,
`sample_ids.tsv`). One config per model is in `paper_pipeline/config/`; the prompts
are in `paper_pipeline/src/paper_pipeline/prompts.py`. Step-by-step server
instructions (setup, a 2-paper test, resuming, packing the results):
`GENERATION_RUN.txt`. All options: `paper_pipeline/README.md`.

## Repository layout

- `data_collection/`: corpus harvesting and text extraction, antithesis
  matching (`antithesis_patterns.py` is the shared pattern list), the annotation
  pool and pages, and the LLM steps (X/Y extraction, stance, straw-man coding).
- `counting/`: analyses behind the paper's tables and figures.
- `paper_pipeline/`: generates, reviews and revises papers with LLMs (dataset 3).
- `data/`: processed data every number is derived from: per-instance pattern
  matches, the annotation pool, labels and codes, stance scores, mined revision
  hunks, each corpus's `sections.jsonl.gz`, and `excluded_documents.tsv`.
- `data/*/text`, `data/*/text_body` (about 1 GB): extracted plain text and body
  text (references and appendices removed), not included; Section 4 regenerates
  them.

**Not included** (too large, and only needed to re-run collection): ACL 2019
PDFs (about 3.2 GB), arXiv LaTeX source (about 19 GB), cloned GitHub repos for
version-history mining (about 1.4 GB), and the repos of the code-release
exploration (about 26 GB, most without a license permitting redistribution;
Section 5).

## Setup

Python 3.10 or newer. The analyses need `numpy scipy pandas pyarrow`; spaCy
(`en_core_web_sm`) for syntax; `langdetect` for the language filter; `openai`
for the LLM steps; `torch transformers` for the aspect-based sentiment check and
the reward models. `paper_pipeline/requirements.txt` covers generation.

The LLM steps read an OpenAI key from a file outside the repo
(`--key-file ../../LYS-API-key.txt`) or, for `paper_pipeline`, from
`OPENAI_API_KEY` or a `.env` file (git-ignored). Never commit a key.

Preference data (Hugging Face, not in the repo): raw UltraFeedback
(`openbmb/UltraFeedback`), HelpSteer2 and HelpSteer3 (`nvidia/HelpSteer2`,
`nvidia/HelpSteer3`), the binarized UltraFeedback preferences
(`uf_train_prefs.parquet`) and the Tülu 3 preference mixture
(`tulu3_*.parquet`). TODO: record the exact dataset ids of the last two.

Commands below assume you are in `data_collection/` or `counting/` as shown.

## 1. Verify the numbers (fast -- uses only what's in this repo)

### Corpus sizes and antithesis-family rates both need body-only text first

Word counts and antithesis rates are computed on each corpus with references and
appendices stripped, so extraction methods that differ in how much back matter they
capture (PDF text vs. LaTeX source) are compared on equivalent content. Run from
`data_collection/`:

```
python3 strip_backmatter.py acl2019 ../data/acl2019/text ../data/acl2019/text_body
python3 strip_backmatter.py arxiv ../data/arxiv2026/text ../data/arxiv2026/text_body
python3 strip_backmatter.py arxiv ../data/arxiv_versions/text ../data/arxiv_versions/text_body
```

See the module docstring for how each corpus's back-matter boundary is detected
(different extraction methods need different strategies) and their success rates.
ACL 2019's short-paper/long-paper breakdown in the corpus-size columns of the Data
table also needs published page counts, from the Anthology export
(`anthology+abstracts.bib.gz`, `pages` field; short = $\leq$7 published pages,
long = $\geq$9), which `corpus_sizes.py --bib` reads.

### Excluded documents (non-English or near-empty text)

Documents that are not usable English text are excluded from every rate, sample
and analysis. The list is checked in as `data/excluded_documents.tsv`; to
regenerate it (about 12 minutes; needs `pip install langdetect`), from
`data_collection/`:

```
python3 language_filter.py --corpus acl2019=../data/acl2019/text_body \
    --corpus arxiv2026=../data/arxiv2026/text_body \
    --corpus arxiv_versions=../data/arxiv_versions/text_body \
    --out ../data/excluded_documents.tsv
```

See the script's docstring for the criteria. Current list: 119 ACL 2019, 20
dataset-(2) and 20 version-diff documents.

### Antithesis-family table / Figures 1-2: rates per 1,000 words

Run from `counting/`, against the body-only text produced above:

```
python3 count_antithesis_patterns.py ../data/acl2019/text_body \
    --exclude ../data/excluded_documents.tsv --exclude-corpus acl2019
python3 count_antithesis_patterns.py ../data/arxiv2026/text_body \
    --exclude ../data/excluded_documents.tsv --exclude-corpus arxiv2026
# v1 vs. latest version (one run per pattern id from antithesis_patterns.py);
# also gives the revision-test table for --pattern rather_than
python3 version_rate_test.py ../data/arxiv_versions/text_body --pattern rather_than \
    --exclude ../data/excluded_documents.tsv
```

`count_antithesis_patterns.py` prints (pattern, total, files-with-a-hit %,
mean/median per 1,000 words); `version_rate_test.py` prints the v1 and latest mean
rates over paper pairs (v1 vs. highest version). These mean rates are what's in the
antithesis-family table and `paper/generate_figures.py`'s `DATA` array (which the
paper's figures are rendered from; update both together if these numbers change).

### Instance counts (appendix: "15,055 (2019), 26,737 (dataset (2)), 10,875 (v1), 12,551 (latest)")

These, and Section 4's high-count-papers analysis below, still run against the full
(reference/appendix-included) `text` directories, not `text_body`. The annotation pool
is therefore drawn from full text; 33 of its 1,000 items lie in back matter, and leaving
them out barely changes the annoying rates (see the paper's pool appendix).

Run from `data_collection/`:

```
python3 extract_antithesis_instances.py ../data/acl2019/text out.jsonl --corpus-name acl2019
python3 extract_antithesis_instances.py ../data/arxiv2026/text out.jsonl --corpus-name arxiv2026
python3 extract_antithesis_instances.py ../data/arxiv_versions/text out.jsonl --glob "*_v1.txt" --corpus-name arxiv_v1
python3 extract_antithesis_instances.py ../data/arxiv_versions/text out.jsonl --glob "*_v[2-9].txt" --corpus-name arxiv_latest
```

Each prints `DONE <n> instances across <k> files -> out.jsonl`; `<n>` is the number
quoted in the paper. (The checked-in `../data/antithesis_instances/*.jsonl` are
already this output -- these commands just let you regenerate and diff them.)

### High-count papers (max count, >=20/>=35 thresholds; top-papers table in additional_experiments.tex)

Run from `counting/`:

```
python3 high_count_report.py ../data/arxiv2026/text --threshold 20 --threshold 35 --top 7
python3 high_count_report.py ../data/acl2019/text --threshold 12 --top 1
```

Reproduces: ACL 2019's corpus-wide max (12), the count of dataset-(2) papers with
>=20 occurrences (124) and >=35 (22; the annotation pool's high-count stratum
threshold -- the stratum's 100 items are drawn from these 22 papers), and
the top-7 list with per-1k rates.

### arXiv version-diff mining (appendix table: hunks added/reworded/removed)

```
python3 arxiv_versions/diff_versions.py ../data/arxiv_versions/text out.jsonl --phrase "rather than"
python3 -c "
import json, collections
c = collections.Counter(json.loads(l)['category'] for l in open('out.jsonl'))
print(dict(c), 'total:', sum(c.values()))
"
```

`category` values: `added`, `reworded` (phrase kept, context reworded), `removed`
(non-surgical), `removed_surgical` (the clause immediately preceding "rather than"
survives verbatim -- see `phrase_diff_utils.py`). `removed from hunk` in the table is
`removed` + `removed_surgical`.

### GitHub version-history mining (additional_experiments.tex)

```
python3 -c "
import json, glob, collections
c = collections.Counter()
for f in glob.glob('../data/github_history/hunks/*.jsonl'):
    for line in open(f):
        c[json.loads(line)['category']] += 1
print(dict(c), 'total:', sum(c.values()))
"
```

(The repos themselves aren't checked in -- see Section 4 to re-mine from scratch.)

### Annotation results (human labels)

Run from `counting/` on `../data/annotation` (labels in `human_labels.jsonl`; see
Section 3 for how they are collected). The paper's annotation tables use both
batches (999 items). `refresh_annotation_numbers.sh` runs everything below in one
go; individually:

```
python3 annotation_report.py ../data/annotation --annotator A1     # rates per stratum, drift
python3 multi_annotator.py ../data/annotation --pair A1 A2   # rates, agreement, band
python3 encounter_model.py ../data/annotation ../data/antithesis_instances \
    --papers acl2019=4863 arxiv2026=1841 --exclude ../data/excluded_documents.tsv
python3 strawman_association.py ../data/annotation   # straw-man table
python3 stance_analysis.py ../data/annotation ../data/annotation/stance/stance__gpt-6-sol.jsonl   # stance table
# --batch N restricts multi_annotator/encounter_model/stance_analysis to one batch
```

Correlates of annoyance (`lexical_contrast.py`, `syntax_features.py`,
`group_cohesion.py`, `semantic_similarity.py`) take `--annotator NAME`; see each
docstring. The paper runs them for each annotator and for the union of both
annotators' labels ("either"), which is a derived label set (an item is annoying if
either annotator labeled it so, legitimate if both did).

### Dataset (3): LLM-written papers

Archives in `data/generated/gpt_papers/` (see its README; unzip first). From `counting/`:

```
python3 generation_rates.py ../data/generated/gpt_papers/batch2 ../data/generated/gpt_papers/batch3 \
    --acl-dir ../data/acl2019/text_body --arxiv-dir ../data/arxiv2026/text_body \
    --excluded ../data/excluded_documents.tsv
```

Prints the papers dropped (excluded list or language filter), the per-source rates
for every pattern, and paired Wilcoxon tests (generated vs. original, revised vs.
generated).

## 2. Regenerate the paper's figures

From the `paper` repo, after updating `generate_figures.py`'s `DATA` array with any
new `count_antithesis_patterns.py` output:

```
python3 generate_figures.py
```

Writes `fig_antithesis_2019_2026.pdf` and `fig_antithesis_v1_latest.pdf`.

## 3. Annotation tool / pool

The annotation pages are static pages on GitHub Pages (repository
`olzama/rather-than-annotator`: `index.html` for the main pool, `#revision` for the
revision set, `strawman/` for straw-man coding, `authorship/` for the authorship-guess task). Labels are saved to a Google Sheet
through a Google Apps Script web app (`data_collection/annotation_tool/sheet_backend.gs`):
every save appends a row (received_at, annotator, item_id, label, client_ts, set),
and the latest row per annotator and item counts. The pages keep unsent labels in the
browser and retry. The authorship-guess page writes to its own sheet (a second deployment of the
same script), so the main sheet stays small; pass that sheet's URL to
`build_authorship_page.py` and `authorship_analysis.py --sheet-url`.

```
# pull labels from the sheet into human_labels.jsonl (revision items go to
# revision/human_labels.jsonl); an annotator's rows are replaced, others kept.
# Only the listed names are pulled, so test names ("zz-test-...") stay out.
python3 pull_sheet_labels.py ../data/annotation --sheet-url <apps-script-url> \
    --annotators A2 A5

# rebuild the main page (then wrap it in a doctype/head and copy it to the
# rather-than-annotator repo as index.html)
python3 annotation_tool/build_page.py ../data/annotation/items_public.jsonl out.html \
    --revision ../data/annotation/revision/items_public.jsonl \
    --sheet-url <apps-script-url> --sheet-token rt-2026-annotate

# rebuild the straw-man page (strawman/index.html in the same repo)
python3 annotation_tool/build_strawman_page.py ../data/annotation/strawman out.html \
    --sheet-url <apps-script-url>

# add a batch of items; new items carry "batch": N, and the page orders and
# repeats each batch separately, so earlier batches are unchanged for annotators
python3 add_annotation_batch.py ../data/antithesis_instances ../data/annotation \
    --corpus acl2019:100 --corpus arxiv2026:238 --corpus arxiv_v1:89 \
    --corpus arxiv_latest:73 --seed 20260926 --batch 2

# build the revision set: instances removed surgically between arXiv v1 and
# the latest version, plus instances from the same papers kept unchanged
python3 revision_sets.py ../data/arxiv_versions/rather_than_hunks.jsonl \
    ../data/antithesis_instances/arxiv_v1.jsonl ../../data/arxiv_versions/text \
    ../data/annotation ../data/annotation/revision

# retire specific pool items and redraw replacements from a (corrected) corpus
python3 retire_and_redraw_items.py ../data/annotation --instances-dir ../data/antithesis_instances \
    --corpus <acl2019|arxiv2026|arxiv_v1|arxiv_latest> --item-ids-file <ids.txt>
    # add --corpus-label high_count_arxiv2026 --min-paper-count 35 --text-dir ../data/arxiv2026/text \
    #     --max-per-paper 10  when redrawing the high-count stratum specifically
```

The pool has two batches of 500 items (`items_public.jsonl`; `items_provenance.jsonl`
holds corpus and document). A provenance row with `"excluded"` (currently item_2209,
from a French paper) stays on the page but is skipped by every analysis.

Hidden repeat items measure annotator drift. The page generates them per
annotator: 40 items from the first 100 in that annotator's order are shown
again at random positions in the second half, and the copies' labels are
saved under `rep__<item_id>`. Each batch has its own 40 repeats. `../data/annotation/repeats.jsonl` lists an
earlier set of 40 fixed copies (item_2148 to item_2187, with `repeat_of` and
the original label), used for the first annotator's batch 1 and no longer part of
the page (the page skips batch-1 repeats for that annotator,
`FIXED_REPEATS_BATCH1`); their provenance rows carry `repeat_of`. `counting/annotation_report.py`
handles both kinds and excludes repeats when computing rates.

## 4. Full pipeline from scratch (slow -- needs raw source not in this repo)

Only needed to re-run *collection* itself (re-download and re-extract), not to
verify the paper's numbers (Section 1 already does that from checked-in data).

**(1) ACL Anthology 2019:**
```
curl -s -o anthology.bib.gz https://aclanthology.org/anthology+abstracts.bib.gz
gunzip -k anthology.bib.gz
python3 extract_acl2019_ids.py anthology.bib > acl2019_ids.txt
./download_acl2019.sh acl2019_ids.txt ../data/acl2019
```

**(2) ACL-style arXiv papers, 2026:**
```
python3 harvest_arxiv2026.py ../data/arxiv2026/arxiv2026_candidates.tsv
python3 build_arxiv2026_corpus.py ../data/arxiv2026/arxiv2026_candidates.tsv ../data/arxiv2026 \
    --reuse-dir ../data/arxiv_versions
```

**arXiv version-diff subset (feeds both dataset-(2)'s `--reuse-dir` above and the version-diff mining table):**
```
python3 arxiv_versions/download_arxiv_versions.py ../data/arxiv2026/arxiv2026_candidates.tsv ../data/arxiv_versions
```
(`build_arxiv2026_corpus.py` and `download_arxiv_versions.py` both redact LaTeX
math/`\ref`/`\cite` to `[redacted]` before running `detex` -- see
`latex_redact.py`'s docstring for why plain `detex` and its `-r` flag are both unsafe
here. `reextract_arxiv_detex.py` re-runs this over already-downloaded sources
without re-fetching, if you're re-applying a pipeline fix rather than starting cold.)

**GitHub version-history mining:**
```
python3 version_history/search_github_repos.py candidates.tsv
python3 version_history/check_commit_history.py candidates.tsv counts.tsv
python3 version_history/mine_edit_history.py counts.tsv ../data/github_history --min-commits 5
```

## 5. Code-release link exploration (not tied to a paper table)

Exploratory follow-up, prompted by "how many papers link to their own code on
GitHub" as opposed to merely citing a github.com URL for a tool they used --
not a number reported in the paper itself.

```
# from counting/: flag papers whose text links to their own GitHub code release
# -- see the script's docstring for the cue-phrase heuristic and the specific
# false-positive shapes it guards against
python3 count_code_release_links.py ../data/acl2019/text --corpus-name acl2019 --jsonl acl2019_code.jsonl
python3 count_code_release_links.py ../data/arxiv2026/text --corpus-name arxiv2026 --jsonl arxiv2026_code.jsonl

# from data_collection/: clone every linked repo (shallow, no GitHub token needed;
# resumable -- already-cloned repos are skipped instantly on a re-run)
python3 download_code_release_repos.py <out_dir> ../counting/acl2019_code.jsonl ../counting/arxiv2026_code.jsonl

# check each cloned repo for a LICENSE file
python3 scan_repo_licenses.py <out_dir>/repos <out_dir>/licenses.json

# join paper text + README + code into one record per paper, split into a full
# local-only artifact and a metadata-only manifest safe to commit
python3 build_code_paper_dataset.py <out_dir>/manifest.jsonl ../data \
    <out_dir>/paper_code_dataset.jsonl \
    --licenses <out_dir>/licenses.json \
    --metadata-out <out_dir>/paper_code_manifest.jsonl
```

As of 2026-09-15 (after the `\input`/`\include` extraction fix documented under
"Known extraction fixes" below, which raised arXiv 2026's own-code-link rate
from 510 to 652 papers, since a code-availability sentence is often near the
end of a paper, exactly where content was going missing): 903 papers (251
ACL 2019, 652 arXiv 2026)
have an apparent own-code link; 832 unique repos, 738 clonable (98 gone/
private, 67 had a truncated URL in the extracted text). Of the 734 scanned,
**345 (47%) have no LICENSE file** -- default copyright, no redistribution
right even though the repo is public on GitHub.

Because of that, only the metadata manifest is checked in, at
`data/code_release_repos/paper_code_manifest.jsonl` (doc_id/repo/license/
file-count facts, no repo content -- see that directory's README for the full
license breakdown). The full joined dataset (paper text + README + code text
verbatim) and the cloned repos themselves are NOT checked in anywhere; they're
a private research artifact on disk, same reasoning as this section's "Not
included" note above.

## 6. Section-level splitting and titles (abstract / introduction / method / ...)

Also not tied to a paper table -- splits each paper's text into top-level
sections, for anything that needs "just the method section" or "just the
abstract" rather than the whole document. Each record is
`{"doc_id", "title", "sections": [{"category", "heading", "text"}, ...], ...}`
plus per-corpus split statistics. The two corpora need genuinely
different approaches, not the same heuristic tuned twice:

```
# arXiv 2026: detex'd LaTeX text has clean, correctly-ordered headings
# (from \section{...} directly), so a whitelist-based text heuristic works.
# --report-unmatched first, to see what a new corpus's alias list is missing.
python3 split_paper_sections.py ../data/arxiv2026/text --jsonl ../data/arxiv2026/sections.jsonl

# ACL 2019: pdftotext's column handling was found (checked by hand across
# several papers) to scramble BOTH heading formatting (a heading's number and
# title can be dozens of lines apart) AND paragraph reading order itself (one
# paper's subsection "3.2" appeared before "3.1" in the extracted text) -- not
# a heading-detection problem a better regex can fix. Needs a real
# scholarly-PDF parser instead: GROBID (built from source with JDK 21, no
# Docker needed -- see tools/grobid/, `./gradlew run`, default port 8070).
python3 process_acl2019_grobid.py ../data/acl2019/pdfs <out_dir>/tei --workers 6
python3 parse_grobid_tei.py <out_dir>/tei --jsonl ../data/acl2019/sections.jsonl

# Compress, then add titles: official Anthology titles for ACL 2019 (from the
# Anthology's per-volume XML, cached in data/acl2019/anthology_xml/, not
# checked in), harvest-TSV titles for arXiv 2026.
gzip -f ../data/arxiv2026/sections.jsonl ../data/acl2019/sections.jsonl
python3 add_titles_to_sections.py ../data
```

As of 2026-09-15: GROBID processed 4,876/4,877 ACL 2019 PDFs successfully;
13 of those are excluded from the corpus (front matter, and one Anthology
entry whose PDF is a different paper -- see `extract_acl2019_ids.py`),
leaving 4,863. Section coverage (papers with >=1 section of that category):
4,863/4,863 abstract (42 of them empty), 4,577 introduction, 4,067 conclusion, 2,115
related_work, 1,895 results, 1,708 experiments, 1,346 method, 1,080
analysis, 972 appendix, 3,288 acknowledgments. (Contrast this with what the plain-text heuristic
manages on arXiv 2026's cleaner source: still only ~49% introduction recall
in an early pass, because even that corpus's heading formatting isn't fully
consistent -- see split_paper_sections.py's docstring.) Both scripts are
conservative about miscategorized content: an unrecognized heading in the
arXiv 2026 heuristic stays embedded in the previous section rather than
creating a wrong boundary; GROBID's `<div>` boundaries are structurally
trustworthy either way, so an unmatched heading there just gets category
`"other"` with the raw heading preserved, not merged into anything.

The ACL 2019 `sections.jsonl` is checked in gzip-compressed, at
`data/acl2019/sections.jsonl.gz` (`gunzip -k` to use). The arXiv 2026 one is not
included; the commands above rebuild it from the downloaded sources. The raw
GROBID TEI XML is not included either (regenerate it from the PDFs).

## Known extraction fixes (why "corrected" appears throughout)

- **ACL 2019**: `pdftotext -layout` interleaves left/right column text on two-column
  PDFs into garbled sentences. Fixed by dropping `-layout` (see `download_acl2019.sh`).
- **arXiv (dataset (2), version-diff subset)**: plain `detex` silently deletes LaTeX
  math and `\ref`/`\cite` commands instead of substituting anything, and its own `-r`
  math-placeholder ("noun") is itself a common word in this corpus (NLP papers).
  Fixed by `latex_redact.py`: `\ref`/`\cite` redacted by regex before `detex` runs;
  math redacted by collapsing `detex`'s own between-word whitespace gaps after.
- **arXiv (dataset (2), version-diff subset)**: `detex` resolves `\input`/`\include`
  relative to its own working directory, not the source file's, so invoking it
  without setting that directory (as the pipeline originally did) silently dropped
  all body content for any paper splitting content across `\input`'d files --
  common practice, not an edge case -- while still exiting 0 (a stderr warning, not
  a nonzero exit code). Affected 774/1,841 (42%) of dataset-(2) papers and 794/
  1,617 (49%) of version-diff files, down to near-empty extractions in the worst
  cases. Fixed in `latex_redact.py`'s `run_detex()` (sets `cwd` to the source
  file's directory).

All three fixes changed corpus text, so every number derived from it (all corpus
rates and sizes, the figures, the annotation pool, and the code-release-link
exploration in Section 5) reflects the corrected extraction as of 2026-09-15.

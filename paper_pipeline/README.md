# paper_pipeline

Generate → review → revise papers from the title and abstract of real
papers, drawn at random from ACL 2019 and arXiv 2026
(`data/acl2019/sections.jsonl.gz`, `data/arxiv2026/sections.jsonl.gz`). Each
generation uses one complete-paper model request containing only fixed
instructions, the source title, and the abstract. The original body sections
and metadata never enter the model prompt. All three stages share the existing
model backend and use their own configured decoding parameters and system
prompts.

## Setup and run

Run from the repository root:

```bash
python3 -m venv paper_pipeline/.venv
source paper_pipeline/.venv/bin/activate
pip install -r paper_pipeline/requirements.txt
# a test on 5 papers, drawn without recording them in the used-papers registry
PYTHONPATH=paper_pipeline/src python3 -m paper_pipeline.sample_ids --n 5 --seed 1 --out /tmp/test_ids.tsv
PYTHONPATH=paper_pipeline/src python3 -m paper_pipeline.main --ids-file /tmp/test_ids.tsv
```

Set `OPENAI_API_KEY` in your environment or `.env` for API models. Local models
require a suitable PyTorch installation and sufficient memory.
`--num-papers N` draws N papers at random from ACL 2019 and arXiv 2026 (see
"Which papers are generated"); omit it to process every valid record. Papers
whose model call fails still count toward N.

Relative input and output paths resolve from the current working directory;
the default config path resolves relative to the package.

```bash
PYTHONPATH=paper_pipeline/src python3 -m paper_pipeline.main \
    --config paper_pipeline/config/config.yaml \
    --batch batch3 --num-papers 100 --seed 20260928
```

Each output directory contains the same filenames, for example
`2019.ccnlg-1.1.md` or `2601.00263.md` (the source paper's id). Generated files contain only model-produced paper text.
All output folders and the log are grouped in one folder named after the configured
model. All three stage folders are created at startup, including for single-stage
runs. For example:

```text
gpt-4o/
├── generated_papers/
├── evaluated_papers/
├── revised_papers/
├── pipeline.log
└── sample_ids.tsv
```

`pipeline.log` appends timestamped messages on each run; logs also appear in the
console. `sample_ids.tsv` is written when selecting papers from configured sources.
Slashes and other unsafe characters in the model folder name become underscores.
The optional `--generated-dir`, `--evaluated-dir`, and `--revised-dir` flags override
individual output paths; omit them to keep everything together as shown above.

## Batch 2 (per-model configs)

`config/gpt-4o.yaml`, `config/gpt-5.6-sol.yaml` and `config/gpt-6-sol.yaml`
set the model and its output limit (16384 tokens) and keep the configured
`sources` and `excluded_documents`. The batch-2 command and server instructions
are in the repository root's `README.md` and `GENERATION_RUN.txt`.

## Which papers are generated

The config's `sources` lists the sections files papers are drawn from
(ACL 2019 and arXiv 2026). `--num-papers N` draws N papers at random, split
evenly across the sources (`--sources acl2019` to use one), reproducible
from `--seed` (default 20260928). Without `--num-papers`, every valid
record of every source not yet used is generated. The papers selected are written to
`<model>/sample_ids.tsv` (`corpus<TAB>doc_id`) before generation starts. To
generate a batch's papers with another model, run it with the same `--batch`;
any `sample_ids.tsv` can also be passed as `--ids-file`:

```bash
PYTHONPATH=paper_pipeline/src python3 -m paper_pipeline.main --batch batch3 --num-papers 100
PYTHONPATH=paper_pipeline/src python3 -m paper_pipeline.main --ids-file gpt-5.5/sample_ids.tsv \
    --config other_model.yaml
# or draw the sample first, without generating:
PYTHONPATH=paper_pipeline/src python3 -m paper_pipeline.sample_ids \
    --n 100 --seed 20260928 --out data/generation/sample_ids.tsv
```

Papers are never drawn twice across batches. `paper_pipeline/used_papers.tsv`
(config key `used_papers`) records every paper ever drawn, one line per paper:
`batch<TAB>corpus<TAB>doc_id`. Drawing papers requires `--batch NAME`: the first
run of a batch draws `--num-papers` papers from those not in the registry and
appends them under NAME; later runs with the same NAME (other models, reruns)
reuse exactly those papers, and a different `--num-papers` is an error. Batches
may differ in size. Commit the registry after each new batch. `--ids-file`
runs name their papers explicitly and do not touch the registry;
`--exclude-ids FILE ...` adds further papers never to draw. `sample_ids`
excludes the registry too, and records its draw only with `--batch`. The
registry is kept separate from `excluded_documents.tsv`, which the corpus
counts and the annotation samplers also read.

Documents listed in `data/excluded_documents.tsv` (non-English or
near-empty text, see `data_collection/language_filter.py`) are never
selected, and `--ids-file` refuses them. The config key
`excluded_documents` sets the file (`--excluded` overrides it); if the
configured file is missing, generation stops with an error.

`--input-jsonl FILE` generates from one explicit file instead, in file
order (the first `--num-papers` valid records); ACL 2019 exclusions apply.

## Input and failures

The reader streams JSONL in file order. Each record needs a non-empty string
`doc_id`, `title`, and an abstract section identified by `category == "abstract"`,
regardless of its position. Multiple non-empty abstract sections are joined in
order. Other sections and metadata are discarded before generation.

Invalid JSON, missing fields, duplicate identifiers, and conflicting output
filenames are logged with line numbers and skipped. The first occurrence reserves
an identifier, even if malformed. Ordinary ACL IDs remain unchanged; unsafe
filename characters are percent-encoded reversibly (including literal percent
signs). Overlong filenames are reported and skipped. Case-insensitive filename
collisions are also skipped for portability.

Each successful stage is saved immediately. A failure in generation, review, or
revision is logged by `doc_id`; later records are still attempted. A run with
stage failures exits nonzero after processing the remaining records. Repeated
runs replace outputs for matching valid IDs; duplicates within a run cannot
replace earlier outputs.

## Independent stages

Append these options to `PYTHONPATH=paper_pipeline/src python3 -m paper_pipeline.main`:

| Task | Options |
| --- | --- |
| Generate only | `--stage generate --batch NAME --num-papers 2` |
| Review a folder | `--stage review --papers-dir <model-name>/generated_papers` |
| Revise a folder | `--stage revise --papers-dir <model-name>/generated_papers --reviews-dir <model-name>/evaluated_papers` |
| Review one paper | `--stage review --paper <model-name>/generated_papers/2019.ccnlg-1.1.md` |
| Revise one paper | `--stage revise --paper <model-name>/generated_papers/2019.ccnlg-1.1.md --review <model-name>/evaluated_papers/2019.ccnlg-1.1.md` |
| Review and revise one paper | `--paper <model-name>/generated_papers/2019.ccnlg-1.1.md` |
| Extract antithesis matches | `--stage extract --papers-dir <model-name>/generated_papers` |

Folder review/revision scans `.md` and legacy `.txt` papers in sorted order,
without recursion. Matching reviews and all new outputs use `<paper-stem>.md`.
`--skip-existing` skips existing review/revision outputs. Pending inputs are
validated before loading a model. Antithesis extraction runs automatically after generation and revision, including
standalone stages and completed outputs from a partially failed run. It scans all
Markdown papers in each applicable output directory and refreshes
`antithesis/antithesis_instances.jsonl`, retaining document IDs, matched patterns,
and text offsets. Reviews are not scanned automatically. Use `--stage extract`
to refresh matches manually without loading a model.

The obsolete `topics` stage, `--topics`, `--num-topics`, `--generated-topics-dir`,
and CLI `--top-k` have been removed. Use `--num-papers` for the input limit;
model decoding `top_k` remains unchanged.

## Configuration and tests

The existing `model`, `quantization`, `generation`, `evaluation`, and `revision`
configuration remains in place. `generation.paper` controls target length and
suggested academic sections. Review headings come from `evaluation.sections`, and
`evaluation.model_name` can point to a different reviewer model than the one used
for generation and revision. Generation and evaluation require
`num_return_sequences: 1`. The backend's provider behavior and token/decoding
handling are unchanged. Each complete paper, review, and revision must fit the
selected model's context window.

```bash
PYTHONPATH=paper_pipeline/src python3 -m unittest discover -s paper_pipeline/tests
```

Tests mock model calls and cover streaming seed validation, abstract selection,
source-content isolation, one call per stage, filenames, duplicates, limits,
failure continuation, configuration, standalone stages, and output overrides.

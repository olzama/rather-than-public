# Human annotation data

Labels are collected on the annotation pages (GitHub Pages, `olzama/rather-than-annotator`)
and stored in a Google Sheet; `data_collection/pull_sheet_labels.py` writes the latest
label per annotator and item into the files below. `rather-than-judgments.csv` is the
raw export of that sheet (downloaded 2026-10-01): every saved row, including superseded
labels, retired items and test rows. The label files keep the latest row per annotator
and item.

## Annotators

The label files identify annotators by their codes in the paper:

| Annotator | Author | Career stage (2026) | Data |
|---|---|---|---|
| A1 | yes | PhD + 5 years | annoyance labels (pool, GPT comparison set); straw-man codes; authorship judgments |
| A2 | yes | PhD + 17 years | annoyance labels (pool, GPT comparison set); straw-man codes; authorship judgments |
| A3 | yes | first-year PhD student | straw-man codes; authorship judgments (pool task) |
| A4 | no | PhD in 2026 | straw-man codes |
| A5 | yes | second-year PhD student | authorship judgments (pool task); 51 annoyance labels and 11 straw-man codes, not used in the paper |

`Either` is not an annotator: some analyses derive it from A1's and A2's labels (an item
is *annoying* if either labeled it so, legitimate if both did). Names starting with
`zz-test` or `test` are test rows and are ignored by every script.

## Files

- `items_public.jsonl`: the 1,000-item pool (two batches of 500), as shown to the
  annotators: sentence, context before and after, position of the construction.
  Hidden repeats have ids starting with `rep__` in the label files.
- `items_provenance.jsonl`: corpus, document and character span of each item; one
  excluded item is marked `"excluded"`.
- `human_labels.jsonl`: annoyance labels for the pool (`annoying`, `legitimate`,
  `unsure`, `garbled`), one row per annotator and item.
- `GUIDELINE.md`: an early version of the instructions. The instructions as shown on
  the annotation page are in the paper's appendix; there, *legitimate* means only
  "does not annoy you".
- `strawman/`: the straw-man coding sets (`dev.jsonl`, `test.jsonl`), the definition
  shown to the coders, and their codes (`human_codes.jsonl`).
- `stance/`: GPT-6-sol stance ratings and the blind human check of inverted items
  (`spot_check/`, answers by A1).
- `xy_alternatives.jsonl`: the affirmed (X) and rejected (Y) alternative of each item,
  extracted by GPT-6-sol.
- `revision/`: the revision item set (instances removed between arXiv versions).

Related folders:

- `../annotation_b3/`: the GPT comparison set (400 items: GPT, arXiv 2026, ACL 2019),
  labeled by A1 and A2 with the same file layout, plus its X/Y, stance and cohesion
  outputs.
- `../authorship/`, `../authorship_pool/`: passages for the authorship-guess tasks;
  their answers go to a separate sheet.

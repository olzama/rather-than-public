# arXiv version-diff mining (supplementary corpus)

A second supplementary source, alongside `../version_history/`, for the
"legitimate vs. annoying antithesis" question: does an author's own
revision (preprint -> later/camera-ready version) keep, add, or cut a
"rather than" construction?

This is the higher-yield sibling of `../version_history/`: instead of
relying on authors happening to maintain a public git history (rare —
most matching GitHub repos turn out to be one-shot uploads, see that
directory's README), we use arXiv's own version history directly. Any
paper posted more than once to arXiv already has a real, author-made
diff between versions, and a version bump very often corresponds to a
preprint -> camera-ready revision for exactly the *ACL-venue papers
`../harvest_arxiv2026.py` already collects.

**Yield, checked 2026-09-11**: of the 1,985 candidates from
`harvest_arxiv2026.py`'s full run (cs.CL papers in 2026 whose arXiv
comment mentions an *ACL venue), roughly 41% have 2+ posted versions —
far more usable than the ~15% GitHub yield.

## Pipeline

```
# 1. Download source for v1 and the latest version of each multi-version
#    candidate, extract the main .tex, and convert to plain text.
python3 download_arxiv_versions.py ../../../data/arxiv2026/arxiv2026_candidates.tsv \
    /mnt/kesha/rather-than/data/arxiv_versions --min-version 2

# 2. Diff v1 against the latest version and extract hunks where the
#    target phrase was added, removed, or reworded.
python3 diff_versions.py /mnt/kesha/rather-than/data/arxiv_versions/text \
    /mnt/kesha/rather-than/data/arxiv_versions/rather_than_hunks.jsonl
```

Stage 1 writes plain text to `<out_dir>/text/<arxiv_id>_v<n>.txt` (only
v1 and the latest version are fetched, not every intermediate version)
and raw downloaded/extracted sources to `<out_dir>/raw/` — point
`<out_dir>` at the shared `data/` folder, not inside this repo.

Stage 2's output records look like:

```json
{"arxiv_id": "...", "v_from": 1, "v_to": 3, "tag": "replace",
 "removed": ["<line(s) from v1>"], "added": ["<line(s) from the later version>"],
 "context_before": ["..."], "context_after": ["..."]}
```

`tag` is `"delete"` (phrase-bearing text removed with nothing added in
its place), `"insert"` (newly added), or `"replace"` (the surrounding
text changed on both sides — read `removed`/`added` together to see
whether the phrase itself was cut or just the sentence around it moved).

## Notes / limitations

- Only v1 vs. the latest version is diffed, not every intermediate step.
  A paper with 3+ versions may have added and then removed the phrase
  across different revisions; this pipeline would only show the net
  effect. Fine for our purposes (legitimate-vs-cut, not full revision
  tracking), but worth remembering when reading results.
- "Latest version at harvest time" (the `version` column from
  `harvest_arxiv2026.py`) can be stale if a paper was updated again
  since harvesting; this undercounts rather than overcounts, so it's a
  conservative simplification.
- Text extraction quality depends on `detex`, same caveats as the main
  corpora (see `../README.md`).

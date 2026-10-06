# Code-release repos: metadata

`paper_code_manifest.jsonl` -- one row per paper (ACL 2019 / arXiv 2026)
whose text apparently links to its own code release on GitHub (see
`../../counting/count_code_release_links.py`), for the repo that was
actually clonable:

```json
{"corpus": ..., "doc_id": ..., "owner_repo": ..., "license": ...,
 "repo_url": ..., "n_files_total": ..., "n_files_included": ...,
 "code_truncated": ..., "total_code_chars": ...}
```

`license` comes from `../../data_collection/scan_repo_licenses.py`
matching each repo's root LICENSE file (or `"NO_LICENSE_FILE"`/
`"unrecognized_license_text"`). As of 2026-09-15 (after the
`\input`/`\include` extraction fix -- see `../../README.md`'s "Known
extraction fixes", which raised arXiv 2026's own-code-link count
substantially), of the 738 papers here: 345 have no LICENSE file at all
(default copyright -- no redistribution right, even though the repo is
public on GitHub), 206 MIT, 123 Apache-2.0, 21 GPL-3.0, 12 CC-BY-NC, 8
BSD-2-Clause, 6 AGPL, 6 CC-BY, 5 unrecognized, 3 BSD-3-Clause, 3 LGPL.

**This file is metadata only** -- doc_id/repo/license/clone stats, no
paper text or repo content, so it's fine to commit regardless of any
individual repo's license. The corresponding full dataset (paper text +
README + code file contents, built by
`../../data_collection/build_code_paper_dataset.py`) is NOT here and
NOT committed anywhere -- given the license mix above, redistributing
that verbatim would exceed what most of these repos actually permit. It
lives locally at `/mnt/kesha/rather-than/data/code_release_repos/
paper_code_dataset.jsonl` as a private research artifact; treat it the
same way (don't commit or redistribute further without checking license
per repo you actually want to use/share).

The repos themselves are cloned to `/mnt/kesha/rather-than/data/
code_release_repos/repos/` (~26GB+, not checked in anywhere, same
convention as `../github_history`'s cloned repos).

## If you've been sent the full dataset (paper_code_dataset.jsonl)

Whoever built it should give you the file directly (it's not in git --
see above), but the tool to use it IS in this repo:
`../../data_collection/code_paper_dataset.py`. Don't write a fresh
JSONL-scanning loop against a file this size -- it builds a `doc_id ->
byte offset` index next to the file on first use (a `<file>.index.json`
sidecar), so single-paper lookups don't mean scanning the whole file,
and it auto-rebuilds that index if it's missing, corrupt, or the
dataset file changed underneath it.

```
# one paper, human-readable summary
python3 code_paper_dataset.py get <path-to>/paper_code_dataset.jsonl D19-1004

# one field's raw text (e.g. for feeding to something else)
python3 code_paper_dataset.py get <path-to>/paper_code_dataset.jsonl D19-1004 --field paper_text

# the whole record as JSON
python3 code_paper_dataset.py get <path-to>/paper_code_dataset.jsonl D19-1004 --full

# browse: doc_id / corpus / owner_repo / license, one per line
python3 code_paper_dataset.py list <path-to>/paper_code_dataset.jsonl --corpus acl2019 --license MIT

# record counts by corpus and by license
python3 code_paper_dataset.py stats <path-to>/paper_code_dataset.jsonl
```

Or as a library, if you're writing your own analysis:

```python
import sys
sys.path.insert(0, "<path-to>/code/data_collection")
from code_paper_dataset import Dataset

ds = Dataset("<path-to>/paper_code_dataset.jsonl")
record = ds.get("D19-1004")                    # -> dict, or None if not found
for record in ds.iter(corpus="arxiv2026"):      # streams; use this over repeated
    ...                                          # .get() calls if you're touching most records
```

Every record's `license` field is the reason this file isn't just handed
out further without thought -- **before quoting, reusing, or
redistributing any individual repo's `readme`/`code_files` content
beyond this research group, check that record's `license`**; roughly
half have `"NO_LICENSE_FILE"`, meaning no redistribution right was
actually granted.

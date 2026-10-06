# Version-history mining (supplementary corpus)

This is a third, much smaller and noisier data source than the ACL 2019
and arXiv 2026 corpora (see `../README.md`). It's for the question raised
in the paper's "Legitimate vs. annoying antithesis" subsection: rather
than only looking at the *final* text of a paper, can we find cases where
an author wrote "rather than" and then either kept it across revisions
(a signal it read as legitimate) or cut it in a later edit (a signal it
read as awkward)? That requires papers written under real version
control with a real edit history — not just a PDF snapshot.

## What this finds, empirically

We prototyped this on 2026-09-11. Searching GitHub code search for
`"\usepackage{acl}" filename:main.tex` returns 68 repositories. Checking
commit counts on that file:

| commits on file | # repos |
|---|---|
| 1 (one-shot upload) | 36 (53%) |
| 2-4 | 22 |
| 5+ | 10 |
| (max seen) | 100 |

So **most matches are dead ends** — someone uploaded a finished paper
once and never touched it again in git. That's expected and not a bug:
most Overleaf users never wire up the git bridge, and most who do use it
as a backup rather than a fine-grained revision log. Only the tail (~15%
of candidates, in this sample) has enough history to be useful.

Across the 8 (of 10) repos where mining found anything, we get 95 diff
hunks touching the phrase:

| category | # hunks | meaning |
|---|---|---|
| `reworded` | 55 | phrase kept, surrounding text changed |
| `added` | 24 | phrase newly introduced |
| `removed` | 10 | phrase cut, but as part of a larger rewrite (the clause before it didn't survive either) |
| `removed_surgical` | 6 | phrase cut with the preceding clause intact -- a clean "X rather than Y" -> "X ..." edit |

Only the 6 `removed_surgical` cases are strong evidence that an author
specifically judged "rather than Y" as the part worth cutting, as opposed
to it disappearing incidentally when a whole paragraph got rewritten for
unrelated reasons. See `../phrase_diff_utils.py` for exactly how that
distinction is drawn (and why it needed two correctness passes to get
right: line-wrap shifts between old/new text in prose source both hid
real matches and produced false "removed" hits when checked per-line
instead of on joined text).

Broader search queries (plain `acl.sty`/`acl_natbib.bst`) return far more
hits (700-1200) but are dominated by AI-agent "paper-writing skill"
template repos that bundle the ACL style files without being papers at
all; `search_github_repos.py` defaults to the `main.tex`-filtered query
specifically to avoid that.

**Conclusion / how to use this**: treat this as a small, hand-checked
supplementary corpus (expect low tens of usable repos, not hundreds),
useful for qualitative examples and case studies, not for anything
requiring statistical power on its own.

## Pipeline

Three stages, each a separate script so partial progress survives interruptions:

```
export GITHUB_TOKEN=...  # classic PAT, no special scopes needed

# 1. Find candidate (repo, file) pairs via GitHub code search.
python3 search_github_repos.py candidates.tsv

# 2. Check how many commits actually touched each file.
python3 check_commit_history.py candidates.tsv commit_counts.tsv

# 3. Clone repos with enough history and mine diffs for the target phrase.
python3 mine_edit_history.py commit_counts.tsv /mnt/kesha/rather-than/data/github_history \
    --min-commits 5 --phrase "rather than"
```

Stage 3 writes cloned repos to `<out_dir>/repos/` and one JSON-lines file
per repo with matching hunks to `<out_dir>/hunks/<org>__<repo>.jsonl`,
each record:

```json
{"repo": "...", "path": "...", "commit": "<sha>", "date": "YYYY-MM-DD",
 "author": "...", "hunk": "<the @@ ... @@ diff hunk text>"}
```

Point `<out_dir>` at the shared `data/` folder, not inside this repo —
stage 3 clones full repo histories, which don't belong in version
control here.

## Notes / limitations

- Stage 1 requires authentication (GitHub code search has no
  unauthenticated tier) and is rate-limited to 10 req/min; the script
  paces itself accordingly.
- `check_commit_history.py` caps at 100 commits/file (one API page) —
  fine for separating "1-shot upload" from "actively edited," not meant
  for precise counts on deeply-edited files.
- This is a comment/filename-based discovery method, not exhaustive: it
  will miss real edited-in-git papers that don't use ACL style, don't
  name their main file `main.tex`, or aren't indexed by GitHub's code
  search (which has its own undocumented coverage limits).

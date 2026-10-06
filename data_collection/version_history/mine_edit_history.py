#!/usr/bin/env python3
"""
For repos with enough edit history (see check_commit_history.py), clone
them and mine the commit-by-commit diff of the target file for hunks that
add or remove a match of --phrase (default "rather than"). This is meant
to build a small, hand-checkable corpus of *legitimate-vs-edited-out*
antithesis usage: cases where an author kept the construction across
revisions (legitimate) vs. cases where they wrote it and then cut it
(a signal, though not proof, that it read as awkward/annoying).

This does not require a GitHub token -- it clones over plain https and
reads local git history.

Usage:
    python3 mine_edit_history.py <in_counts.tsv> <out_dir> [--min-commits 5] [--phrase "rather than"]

<in_counts.tsv> is the output of check_commit_history.py. Repos below
--min-commits are skipped. For each repo above the threshold, clones (or
reuses an existing clone) into <out_dir>/repos/<org>__<name>, and writes
one JSON-lines file per repo into <out_dir>/hunks/<org>__<name>.jsonl,
one record per matching diff hunk:

    {"repo": ..., "path": ..., "commit": ..., "date": ..., "author": ...,
     "hunk": "<the @@ ... @@ hunk text, including +/- lines>",
     "category": "added" | "removed_surgical" | "removed" | "reworded",
     "old_text": "<joined pre-edit text of the hunk>",
     "new_text": "<joined post-edit text of the hunk>"}

See ../phrase_diff_utils.py for what "category" means, in particular the
removed_surgical vs. removed distinction: the former is a clean "X rather
than Y" -> "X ..." edit (the clause before the phrase survives verbatim),
the latter is the phrase disappearing incidentally amid a larger rewrite
of the surrounding text.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phrase_diff_utils import classify_change, join_lines  # noqa: E402

COMMIT_MARK = "@@@COMMIT@@@"


def clone_or_update(repo, repos_dir):
    org_name = repo.replace("/", "__")
    dest = repos_dir / org_name
    if dest.exists():
        return dest
    url = f"https://github.com/{repo}.git"
    print(f"cloning {repo}...", file=sys.stderr)
    result = subprocess.run(
        ["git", "clone", "--quiet", url, str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"CLONE_FAILED {repo}: {result.stderr.strip()[:300]}", file=sys.stderr)
        return None
    return dest


def iter_commit_diffs(repo_dir, path):
    """Yield (commit_hash, date, author, diff_text) for each commit touching path."""
    out = subprocess.run(
        [
            "git", "-C", str(repo_dir), "log", "--follow", "-p",
            f"--format={COMMIT_MARK}%H|%ad|%an", "--date=short", "--", path,
        ],
        capture_output=True, text=True,
    ).stdout
    for chunk in out.split(COMMIT_MARK)[1:]:
        header, _, diff = chunk.partition("\n")
        parts = header.split("|", 2)
        if len(parts) != 3:
            continue
        commit_hash, date, author = parts
        yield commit_hash, date, author, diff


def extract_matching_hunks(diff_text, phrase_re):
    """
    Split a diff into @@ ... @@ hunks and keep the ones where the phrase
    appears in the old side and/or the new side of the hunk, joining each
    side's lines first so a phrase split across a line-wrap isn't missed
    (and isn't mistaken for a removal -- see phrase_diff_utils.py).
    """
    hunks = re.split(r"(?=^@@ )", diff_text, flags=re.M)
    for hunk in hunks:
        if not hunk.startswith("@@"):
            continue
        lines = hunk.splitlines()
        # Include context lines (unchanged, present on both sides) so that
        # non-contiguous removed/added regions in the same hunk don't get
        # concatenated directly onto each other -- that would corrupt the
        # "words immediately before the phrase" snippet used downstream.
        old_lines = [l[1:] for l in lines if (l.startswith("-") and not l.startswith("---")) or l.startswith(" ")]
        new_lines = [l[1:] for l in lines if (l.startswith("+") and not l.startswith("+++")) or l.startswith(" ")]
        old_text = join_lines(old_lines)
        new_text = join_lines(new_lines)
        if phrase_re.search(old_text) or phrase_re.search(new_text):
            yield hunk, old_text, new_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("in_path")
    p.add_argument("out_dir")
    p.add_argument("--min-commits", type=int, default=5)
    p.add_argument("--phrase", default="rather than")
    args = p.parse_args()

    phrase_re = re.compile(re.escape(args.phrase), re.I)

    out_dir = Path(args.out_dir)
    repos_dir = out_dir / "repos"
    hunks_dir = out_dir / "hunks"
    repos_dir.mkdir(parents=True, exist_ok=True)
    hunks_dir.mkdir(parents=True, exist_ok=True)

    with open(args.in_path) as f:
        rows = [l.rstrip("\n").split("\t") for l in f.readlines()[1:] if l.strip()]

    candidates = [(repo, path) for repo, path, n in rows if n.lstrip("-").isdigit() and int(n) >= args.min_commits]
    print(f"{len(candidates)}/{len(rows)} candidates have >= {args.min_commits} commits", file=sys.stderr)

    total_hunks = 0
    for repo, path in candidates:
        repo_dir = clone_or_update(repo, repos_dir)
        if repo_dir is None:
            continue
        records = []
        for commit_hash, date, author, diff in iter_commit_diffs(repo_dir, path):
            for hunk, old_text, new_text in extract_matching_hunks(diff, phrase_re):
                category = classify_change(old_text, new_text, phrase_re)
                records.append({
                    "repo": repo, "path": path, "commit": commit_hash,
                    "date": date, "author": author, "hunk": hunk,
                    "category": category, "old_text": old_text, "new_text": new_text,
                })
        if records:
            out_file = hunks_dir / f"{repo.replace('/', '__')}.jsonl"
            import json
            with open(out_file, "w") as out:
                for r in records:
                    out.write(json.dumps(r) + "\n")
            total_hunks += len(records)
            print(f"{repo}: {len(records)} matching hunks -> {out_file}", file=sys.stderr)
        else:
            print(f"{repo}: no matches", file=sys.stderr)

    print(f"DONE {total_hunks} total matching hunks across {len(candidates)} repos", file=sys.stderr)


if __name__ == "__main__":
    main()

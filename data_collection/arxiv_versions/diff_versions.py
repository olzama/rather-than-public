#!/usr/bin/env python3
"""
Diff the v1 and latest-version plain text of each paper downloaded by
download_arxiv_versions.py, and extract every diff hunk where a target
phrase (default "rather than") was added, removed, or reworded between
the preprint and the later (often camera-ready) version.

Output format mirrors ../version_history/mine_edit_history.py's hunks,
for consistency: one JSON-lines file, one record per matching hunk.

Usage:
    python3 diff_versions.py <text_dir> <out.jsonl> [--phrase "rather than"]

<text_dir> is the `text/` directory produced by download_arxiv_versions.py
(files named <id>_v<n>.txt). For each id with a v1 file and a v<latest>
file (latest != 1), computes a unified diff and looks for hunks whose
changed lines match --phrase.
"""
import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phrase_diff_utils import classify_change, join_lines  # noqa: E402


def group_versions(text_dir):
    by_id = defaultdict(dict)
    for f in text_dir.glob("*_v*.txt"):
        m = re.match(r"^(.+)_v(\d+)\.txt$", f.name)
        if not m:
            continue
        aid, ver = m.group(1), int(m.group(2))
        by_id[aid][ver] = f
    return by_id


def matching_hunks(v1_lines, vlast_lines, phrase_re, context=2):
    """
    Diff at line granularity to localize changes, but check for the phrase
    (and classify the change) on *joined* text per side -- detex output
    line-wraps at whatever column the source .tex happened to break on,
    which commonly shifts across a revision even where the prose itself
    is unchanged, so a naive per-line phrase check both misses matches
    split across a shifted line break and can wrongly call a kept phrase
    "removed". See ../phrase_diff_utils.py.
    """
    sm = difflib.SequenceMatcher(a=v1_lines, b=vlast_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ctx_before = v1_lines[max(0, i1 - context):i1]
        ctx_after = v1_lines[i2:i2 + context]
        removed = v1_lines[i1:i2]
        added = vlast_lines[j1:j2]
        old_text = join_lines(ctx_before + removed + ctx_after)
        new_text = join_lines(ctx_before + added + ctx_after)
        if not (phrase_re.search(old_text) or phrase_re.search(new_text)):
            continue
        category = classify_change(old_text, new_text, phrase_re)
        yield {
            "tag": tag,
            "category": category,
            "removed": removed,
            "added": added,
            "context_before": ctx_before,
            "context_after": ctx_after,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text_dir")
    p.add_argument("out_path")
    p.add_argument("--phrase", default="rather than")
    args = p.parse_args()

    phrase_re = re.compile(re.escape(args.phrase), re.I)
    text_dir = Path(args.text_dir)
    by_id = group_versions(text_dir)

    pairs = []
    for aid, versions in by_id.items():
        if 1 in versions and len(versions) >= 2:
            latest = max(v for v in versions if v != 1)
            pairs.append((aid, 1, latest, versions[1], versions[latest]))

    print(f"{len(pairs)} papers with both a v1 and a later version", file=sys.stderr)

    total_hunks = 0
    with open(args.out_path, "w") as out:
        for aid, v1, vlast, v1_path, vlast_path in pairs:
            v1_lines = v1_path.read_text(errors="replace").splitlines()
            vlast_lines = vlast_path.read_text(errors="replace").splitlines()
            for h in matching_hunks(v1_lines, vlast_lines, phrase_re):
                record = {"arxiv_id": aid, "v_from": v1, "v_to": vlast, **h}
                out.write(json.dumps(record) + "\n")
                total_hunks += 1

    print(f"DONE {total_hunks} matching hunks across {len(pairs)} papers -> {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

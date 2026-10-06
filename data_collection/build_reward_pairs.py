#!/usr/bin/env python3
r"""
Build minimal pairs for the reward-model test: each annotation-pool item's
sentence as written, and the same sentence with the "rather than Y" clause
deleted (Y from xy_alternatives.jsonl, which must follow "rather than"
directly in the sentence). Items where the clause cannot be deleted cleanly
are skipped and counted.

Deletion: "X rather than Y" -> "X"; a sentence-initial "Rather than Y, X"
-> "X" (first letter capitalized); a clause set off by commas
(", rather than Y,") loses both commas. Spacing before punctuation is
tidied.

Output JSONL, one line per item: item_id, corpus, context_before, original,
deleted, clause (the deleted text).

Usage:
    python3 build_reward_pairs.py ../data/annotation ../data/annotation/xy_alternatives.jsonl \
        ../data/reward_pairs/pairs.jsonl
"""
import argparse
import json
import re
from pathlib import Path


def delete_clause(s, rt_start, rt_end, y):
    """Sentence without "rather than Y", or None if Y does not follow directly."""
    after = s[rt_end:]
    m = re.match(r"\s*", after)
    if not after[m.end():].startswith(y):
        return None, None
    y_end = rt_end + m.end() + len(y)
    lead = s[:rt_start]
    if not re.search(r"[A-Za-z0-9]", lead):
        rest = re.sub(r"^\s*,?\s*", "", s[y_end:])
        if not rest:
            return None, None
        return lead + rest[0].upper() + rest[1:], s[rt_start:y_end]
    start = rt_start
    while start > 0 and s[start - 1] == " ":
        start -= 1
    comma_before = start > 0 and s[start - 1] == ","
    tail = s[y_end:]
    if comma_before and tail.startswith(","):
        start -= 1
        tail = tail[1:]
    out = s[:start] + tail
    out = re.sub(r"\s+([.,;:!?)])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out, s[start:y_end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("xy")
    ap.add_argument("out")
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    pub = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl"))}
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl"))
            if "repeat_of" not in r and "excluded" not in r}
    xy = {r["item_id"]: r for r in map(json.loads, open(args.xy)) if r.get("y")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    with open(args.out, "w") as f:
        for i in sorted(xy):
            if i not in pub or i not in prov:
                continue
            it = pub[i]
            deleted, clause = delete_clause(it["sentence"], it["local_start"], it["local_end"], xy[i]["y"])
            if deleted is None:
                skipped += 1
                continue
            f.write(json.dumps({"item_id": i, "corpus": prov[i]["corpus"], "context_before": it.get("context_before", ""),
                                "original": it["sentence"], "deleted": deleted, "clause": clause}) + "\n")
            n += 1
    print(f"wrote {n} pairs to {args.out}; skipped {skipped} (Y not directly after 'rather than')")


if __name__ == "__main__":
    main()

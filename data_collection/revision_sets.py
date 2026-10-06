#!/usr/bin/env python3
r"""
Build the revision annotation set: "rather than" instances that authors
removed surgically between arXiv v1 and the latest version, and instances
from the same papers that survived revision unchanged.

  removed  v1 instances inside a "removed_surgical" hunk
           (rather_than_hunks.jsonl) whose sentence no longer appears in
           the latest version
  kept     v1 instances from the same papers whose sentence appears
           verbatim (whitespace-normalized) in the latest version; up to
           as many per paper as that paper has removed instances, drawn
           at random

Instances already in the main annotation pool are left out. Writes, to
<out_dir>: items_public.jsonl (what the annotator sees; item ids rev_NNNN,
no condition), items_provenance.jsonl (condition, paper, v1 offsets), and
reports how many hunks could not be matched to an instance.

Usage:
    python3 revision_sets.py <hunks.jsonl> <arxiv_v1 instances.jsonl> <arxiv_versions/text> \
        ../data/annotation <out_dir> [--seed 1]
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

VERSION_RE = re.compile(r"_v(\d+)$")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hunks")
    ap.add_argument("v1_instances")
    ap.add_argument("versions_text_dir")
    ap.add_argument("annotation_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ann = Path(args.annotation_dir)

    pool_keys = set()
    pub = {json.loads(l)["item_id"]: json.loads(l) for l in open(ann / "items_public.jsonl")}
    for l in open(ann / "items_provenance.jsonl"):
        p = json.loads(l)
        if p["item_id"] in pub:
            pool_keys.add((VERSION_RE.sub("", p["doc_id"]), norm(pub[p["item_id"]]["sentence"]), pub[p["item_id"]]["local_start"]))

    inst = defaultdict(list)
    for l in open(args.v1_instances):
        r = json.loads(l)
        if r["pattern_id"] == "rather_than":
            inst[VERSION_RE.sub("", r["doc_id"])].append(r)

    text_dir = Path(args.versions_text_dir)
    latest_text = {}

    def latest(paper):
        if paper not in latest_text:
            files = sorted(text_dir.glob(f"{paper}_v*.txt"), key=lambda f: int(VERSION_RE.search(f.stem).group(1)))
            latest_text[paper] = norm(files[-1].read_text(errors="replace")) if len(files) > 1 else ""
        return latest_text[paper]

    def in_pool(r):
        return (VERSION_RE.sub("", r["doc_id"]), norm(r["sentence"]), r["local_start"]) in pool_keys

    removed, unmatched = [], 0
    seen = set()
    for h in (json.loads(l) for l in open(args.hunks)):
        if h["category"] != "removed_surgical":
            continue
        paper, old = h["arxiv_id"], norm(" ".join(h["removed"]))
        cands = [r for r in inst.get(paper, [])
                 if norm(r["sentence"]) in old and norm(r["sentence"]) not in latest(paper)]
        if not cands:
            unmatched += 1
            continue
        for r in cands:
            k = (r["doc_id"], r["char_start"])
            if k not in seen and not in_pool(r):
                seen.add(k)
                removed.append(r)

    kept = []
    per_paper = defaultdict(int)
    for r in removed:
        per_paper[VERSION_RE.sub("", r["doc_id"])] += 1
    for paper, n in sorted(per_paper.items()):
        lt = latest(paper)
        cands = [r for r in inst[paper] if (r["doc_id"], r["char_start"]) not in seen and not in_pool(r)
                 and len(norm(r["sentence"])) > 30 and norm(r["sentence"]) in lt]
        kept += rng.sample(cands, min(n, len(cands)))

    items = [("removed", r) for r in removed] + [("kept", r) for r in kept]
    rng.shuffle(items)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items_public.jsonl", "w") as fp, open(out / "items_provenance.jsonl", "w") as fv:
        for n, (cond, r) in enumerate(items, 1):
            iid = f"rev_{n:04d}"
            fp.write(json.dumps({"item_id": iid, "sentence": r["sentence"], "local_start": r["local_start"],
                                 "local_end": r["local_end"], "context_before": r.get("context_before", ""),
                                 "context_after": r.get("context_after", "")}) + "\n")
            fv.write(json.dumps({"item_id": iid, "condition": cond, "doc_id": r["doc_id"],
                                 "char_start": r["char_start"], "char_end": r["char_end"]}) + "\n")
    print(f"removed {len(removed)}, kept {len(kept)} (papers: {len(per_paper)}); "
          f"surgical hunks not matched to an instance: {unmatched} -> {out}")


if __name__ == "__main__":
    main()

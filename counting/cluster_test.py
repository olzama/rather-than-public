#!/usr/bin/env python3
r"""
Clustering of annoying labels by paper (the "Annoyance clusters by paper"
analysis in additional_experiments.tex). The straw-man table itself comes
from strawman_association.py.

1. For each annotator: among arXiv 2026 papers with two or more pool items,
   the number of same-paper pairs both labeled annoying, against 5,000
   shuffles of labels within stratum (one-sided p).
2. For each annotator: annoying rate of items whose paper has another item
   coded a straw man by --coder, vs. papers whose coded items include none
   (Fisher's exact test).

Usage:
    python3 cluster_test.py ../data/annotation [--annotators A1 A2] [--coder A1] [--batch 1]
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ARXIV = {"arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    ap.add_argument("--coder", default="A1")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--perms", type=int, default=5000)
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl"))
            if r["corpus"] in ARXIV and "repeat_of" not in r and "excluded" not in r}
    pub = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl"))}
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        labels[r["annotator"]][r["item_id"]] = r["label"]
    codes = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "strawman" / "human_codes.jsonl")):
        codes[r["annotator"]][r["item_id"]] = r["code"]
    paper = lambda i: re.sub(r"_v\d+$", "", prov[i]["doc_id"])
    rng = np.random.default_rng(1)

    print("1. same-paper annoying pairs (papers with >= 2 items)")
    for name in args.annotators:
        items = [i for i in labels[name] if i in prov and i in pub and labels[name][i] != "garbled"
                 and (args.batch is None or pub[i].get("batch", 1) == args.batch)]
        byp = collections.defaultdict(list)
        for i in items:
            byp[paper(i)].append(i)
        multi = [v for v in byp.values() if len(v) >= 2]
        y = {i: labels[name][i] == "annoying" for i in items}
        stat = lambda yy: sum(sum(yy[i] for i in v) * (sum(yy[i] for i in v) - 1) / 2 for v in multi)
        strata = collections.defaultdict(list)
        for i in items:
            strata[prov[i]["corpus"]].append(i)
        null = []
        for _ in range(args.perms):
            yy = {}
            for v in strata.values():
                lab = [y[i] for i in v]
                rng.shuffle(lab)
                yy.update(zip(v, lab))
            null.append(stat(yy))
        null = np.array(null)
        obs = stat(y)
        print(f"  {name}: {len(multi)} papers; pairs {obs:.0f} vs {null.mean():.1f} expected, p = {(null >= obs).mean():.4f}")

        sm = [(any(codes[args.coder].get(j) == "strawman" for j in v if j != i), y[i])
              for v in multi for i in v if any(j in codes[args.coder] for j in v if j != i)]
        a = [b for o, b in sm if o]
        n = [b for o, b in sm if not o]
        if a and n:
            p = stats.fisher_exact([[sum(a), len(a) - sum(a)], [sum(n), len(n) - sum(n)]])[1]
            print(f"     straw man ({args.coder}) elsewhere in paper: {sum(a)}/{len(a)} vs {sum(n)}/{len(n)}, Fisher p = {p:.3g}")


if __name__ == "__main__":
    main()

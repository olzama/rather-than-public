#!/usr/bin/env python3
r"""
Density of the construction and annoyance (arXiv 2026 pool items).

1. Shown text (the sentence and its neighbors): annoying rate of items whose
   shown text has another "rather than", or another antithesis pattern
   (antithesis_patterns.py, "in contrast" left out), vs. items without
   (Fisher's exact test).
2. Whole paper: the paper's "rather than" rate per 1,000 words of body text;
   logistic regression of annoying on log(1 + rate) with stratum fixed
   effects (Wald p).
3. Straw man elsewhere in the paper: for each annotator other than --coder,
   a logistic regression of annoying on whether another item of the paper
   was coded a straw man by --coder, controlling for --coder's label on the
   item, log(1 + paper rate) and the high-count stratum, with standard
   errors clustered by paper. Items whose paper has another coded item.
Garbled items are left out.

Usage:
    python3 density_test.py ../data/annotation --text-dir ../data/arxiv2026/text_body \
        [--annotators A1 A2] [--coder A1]
"""
import argparse
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import PATTERNS

ARXIV = {"arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"}
RT = re.compile(r"\brather\s+than\b", re.I)
OTHER = [re.compile(rx, re.I) for pid, _, _, rx, _ in PATTERNS if pid not in ("rather_than", "in_contrast")]
WORD = re.compile(r"\w+")


def logistic(X, y, groups=None, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
        b += np.linalg.solve(H, X.T @ (y - p))
    Hi = np.linalg.inv(H)
    if groups is None:
        return b, np.sqrt(np.diag(Hi))
    p = 1 / (1 + np.exp(-X @ b))
    meat = np.zeros_like(H)
    for g in set(groups):
        idx = [k for k, gg in enumerate(groups) if gg == g]
        s = X[idx].T @ (y[idx] - p[idx])
        meat += np.outer(s, s)
    G = len(set(groups))
    V = Hi @ meat @ Hi * G / (G - 1)
    return b, np.sqrt(np.diag(V))


def pval(b, se):
    return math.erfc(abs(b / se) / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    ap.add_argument("--coder", default="A1")
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl"))
            if r["corpus"] in ARXIV and "repeat_of" not in r and "excluded" not in r}
    pub = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl")) if r["item_id"] in prov}
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        labels[r["annotator"]][r["item_id"]] = r["label"]
    codes = {r["item_id"]: r["code"] for r in map(json.loads, open(d / "strawman" / "human_codes.jsonl"))
             if r["annotator"] == args.coder}
    paper = lambda i: re.sub(r"v\d+$", "", re.sub(r"_v\d+$", "", prov[i]["doc_id"]))
    rate = {}
    for i in pub:
        p = paper(i)
        if p not in rate:
            f = Path(args.text_dir) / f"{p}.txt"
            t = f.read_text(errors="ignore") if f.exists() else ""
            n = len(WORD.findall(t))
            rate[p] = 1000 * len(RT.findall(t)) / n if n else None
    shown = {i: " ".join([pub[i].get("context_before", ""), pub[i]["sentence"], pub[i].get("context_after", "")]) for i in pub}
    feats = {"another rather than": lambda i: len(RT.findall(shown[i])) > 1,
             "another antithesis pattern": lambda i: any(rx.search(shown[i]) for rx in OTHER)}
    strata = sorted({prov[i]["corpus"] for i in pub})

    for name in args.annotators:
        L = labels[name]
        items = [i for i in pub if L.get(i) not in (None, "garbled") and rate[paper(i)] is not None]
        y = {i: L[i] == "annoying" for i in items}
        print(f"\n== {name}: {len(items)} items, {sum(y.values())} annoying")
        for fname, f in feats.items():
            w = [y[i] for i in items if f(i)]; wo = [y[i] for i in items if not f(i)]
            p = stats.fisher_exact([[sum(w), len(w) - sum(w)], [sum(wo), len(wo) - sum(wo)]])[1]
            print(f"  {fname:28s} {100 * sum(w) / len(w):.1f}% of {len(w)} vs {100 * sum(wo) / len(wo):.1f}% of {len(wo)}; Fisher p = {p:.3g}")
        X = np.array([[1, math.log1p(rate[paper(i)])] + [prov[i]["corpus"] == s for s in strata[1:]] for i in items], float)
        b, se = logistic(X, np.array([y[i] for i in items], float))
        ra = [rate[paper(i)] for i in items if y[i]]; ro = [rate[paper(i)] for i in items if not y[i]]
        print(f"  paper rate per 1k words: median annoying {statistics.median(ra):.2f} vs other {statistics.median(ro):.2f}; "
              f"logistic coef {b[1]:+.2f} (p = {pval(b[1], se[1]):.3g})")
        if name == args.coder:
            continue
        byp = collections.defaultdict(list)
        for i in items:
            byp[paper(i)].append(i)
        rows = []
        for i in items:
            others = [j for j in byp[paper(i)] if j != i and j in codes]
            if not others or labels[args.coder].get(i) in (None, "garbled"):
                continue
            rows.append((paper(i), any(codes[j] == "strawman" for j in others), labels[args.coder][i] == "annoying",
                         math.log1p(rate[paper(i)]), prov[i]["corpus"] == "high_count_arxiv2026", y[i]))
        X = np.array([[1, r[1], r[2], r[3], r[4]] for r in rows], float)
        b, se = logistic(X, np.array([r[5] for r in rows], float), groups=[r[0] for r in rows])
        sm = [r[5] for r in rows if r[1]]; nsm = [r[5] for r in rows if not r[1]]
        print(f"  straw man ({args.coder}) elsewhere in paper: {sum(sm)}/{len(sm)} vs {sum(nsm)}/{len(nsm)}; "
              f"controlled OR {math.exp(b[1]):.2f} (p = {pval(b[1], se[1]):.2f}), {args.coder}'s label p = {pval(b[2], se[2]):.2g}, "
              f"paper rate p = {pval(b[3], se[3]):.2g} ({len(set(r[0] for r in rows))} papers, {len(rows)} items)")


if __name__ == "__main__":
    main()

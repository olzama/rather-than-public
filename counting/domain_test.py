#!/usr/bin/env python3
r"""
Does the paper's research domain relate to annoyance? Tests the Discussion
impression that uses in papers on other scientific domains read as more
legitimate. Specified before the domain labels were produced:

  Items     arXiv 2026 pool items (dataset (2), v1, latest, high-count),
            hidden repeats and excluded items left out, items the
            annotator labeled garbled left out. ACL 2019 has no annoying
            items and is not used.
  Domain    paper_domain.py's label for the item's paper; "other" =
            applied_domain + linguistics.
  Primary   for each label set (each annotator; Either = annoying if either
            annotator labeled it so, among items both labeled): annoying
            rate in core_nlp vs. other papers. Logistic regression
            annoying ~ other + stratum fixed effects; the hypothesis is
            directional (other < core_nlp). Fisher's exact test (two-sided)
            on the pooled 2x2 is also reported.
  Secondary the three domains separately (descriptive only).

--batch restricts to one annotation batch (the paper's annotator tables
use batch 1).

Usage:
    python3 domain_test.py ../data/annotation ../data/annotation/paper_domains.jsonl \
        --pair A1 A2 [--batch 1]
"""
import argparse
import collections
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

ARXIV = ["arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"]
VERSION_RE = re.compile(r"_v\d+$")


def logistic(X, y, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        b += np.linalg.solve(H, X.T @ (y - p))
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("domains")
    ap.add_argument("--pair", nargs=2, default=["A1", "A2"])
    ap.add_argument("--batch", type=int)
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl"))
            if r["corpus"] in ARXIV and "repeat_of" not in r and "excluded" not in r}
    pub = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl"))}
    items = {i for i in prov if i in pub and (args.batch is None or pub[i].get("batch", 1) == args.batch)}
    domain = {r["doc_id"]: r["domain"] for r in map(json.loads, open(args.domains))}
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        if r["item_id"] in items:
            labels[r["annotator"]][r["item_id"]] = r["label"]
    a, b = args.pair
    labels["Either"] = {i: ("annoying" if "annoying" in (labels[a][i], labels[b][i]) else "other")
                        for i in labels[a] if i in labels[b] and "garbled" not in (labels[a][i], labels[b][i])}

    def dom(i):
        return domain.get(VERSION_RE.sub("", prov[i]["doc_id"]))

    unlabeled = sum(dom(i) is None for i in items)
    print(f"items: {len(items)} arXiv 2026{'' if args.batch is None else f' (batch {args.batch})'}; "
          f"without a domain label: {unlabeled}")
    print("papers by domain:", dict(collections.Counter(domain.values())))
    for name in (a, b, "Either"):
        rows = [(i, lab) for i, lab in labels[name].items() if lab != "garbled" and dom(i)]
        by = collections.defaultdict(lambda: [0, 0])
        for i, lab in rows:
            by[dom(i)][0] += lab == "annoying"
            by[dom(i)][1] += 1
        core = by["core_nlp"]
        other = [by["applied_domain"][0] + by["linguistics"][0], by["applied_domain"][1] + by["linguistics"][1]]
        print(f"\n{name}: " + "; ".join(f"{k} {v[0]}/{v[1]} ({v[0] / v[1]:.1%})" for k, v in sorted(by.items()) if v[1]))
        print(f"  core_nlp {core[0]}/{core[1]} ({core[0] / core[1]:.1%}) vs other {other[0]}/{other[1]} "
              f"({other[0] / other[1]:.1%}); Fisher p = "
              f"{stats.fisher_exact([[core[0], core[1] - core[0]], [other[0], other[1] - other[0]]])[1]:.3g}")
        X = np.array([[1.0, float(dom(i) != "core_nlp")] + [float(prov[i]["corpus"] == s) for s in ARXIV[1:]]
                      for i, _ in rows])
        y = np.array([float(lab == "annoying") for _, lab in rows])
        keep = X[:, 2:].sum(axis=0) > 0
        X = np.hstack([X[:, :2], X[:, 2:][:, keep]])
        coef, se = logistic(X, y)
        z = coef[1] / se[1]
        print(f"  logistic, stratum fixed effects: odds ratio other vs core_nlp {math.exp(coef[1]):.2f} "
              f"(95% CI {math.exp(coef[1] - 1.96 * se[1]):.2f}-{math.exp(coef[1] + 1.96 * se[1]):.2f}), "
              f"one-sided p (other < core) = {stats.norm.cdf(z):.3g}")


if __name__ == "__main__":
    main()

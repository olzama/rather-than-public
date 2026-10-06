#!/usr/bin/env python3
r"""
Carry-over in annotation order: are items labeled annoying more often right
after the annotator met an egregious straw man? Items were shown in random
order, so the position of the egregious items is random.

For each annotator, items are ordered by label timestamp (hidden repeats
included, since they were seen). For every arXiv 2026 item (the outcome;
ACL 2019 items have no annoying labels) we record whether one of the
previous --window items shown was (a) an egregious straw man
(EGREGIOUS: the items both straw-man coders coded as straw men), or
(b) an item the annotator labeled annoying. The annoying rate of outcome
items is compared with and without each; logistic regression with both
predictors and stratum fixed effects.

Usage:
    python3 carryover_test.py ../data/annotation [--window 5] [--annotators A1 A2]
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

EGREGIOUS = {"item_1764", "item_1792", "item_1809", "item_1856", "item_1880", "item_2037"}
ARXIV = ["arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"]


def logistic(X, y, iters=100):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
        b += np.linalg.solve(H, X.T @ (y - p))
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl"))}
    rows_by = collections.defaultdict(list)
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        if r.get("ts") is not None:
            rows_by[r["annotator"]].append(r)
    for name in args.annotators:
        seq = sorted(rows_by[name], key=lambda r: r["ts"])
        base = lambda i: i[5:] if i.startswith("rep__") else i
        out = []
        for k, r in enumerate(seq):
            i = base(r["item_id"])
            if i not in prov or "excluded" in prov[i] or prov[i]["corpus"] not in ARXIV or r["label"] == "garbled":
                continue
            prev = seq[max(0, k - args.window):k]
            egr = any(base(p["item_id"]) in EGREGIOUS for p in prev)
            ann = any(p["label"] == "annoying" for p in prev)
            out.append((r["label"] == "annoying", egr, ann, prov[i]["corpus"], i in EGREGIOUS))
        out = [o for o in out if not o[4]]
        y = np.array([o[0] for o in out], float)
        print(f"\n{name}: {len(seq)} labels in order; {len(out)} arXiv 2026 outcome items (egregious items themselves left out); "
              f"egregious items met: {sum(base(r['item_id']) in EGREGIOUS for r in seq)}")
        for j, lab in ((1, "egregious straw man"), (2, "any annoying item")):
            m = np.array([o[j] for o in out])
            k1, n1, k0, n0 = int(y[m].sum()), int(m.sum()), int(y[~m].sum()), int((~m).sum())
            p = stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])[1] if n1 else float("nan")
            print(f"  after {lab} in previous {args.window}: {k1}/{n1} ({k1 / max(n1, 1):.0%}) vs otherwise "
                  f"{k0}/{n0} ({k0 / max(n0, 1):.0%}); Fisher p = {p:.3g}")
        X = np.column_stack([np.ones(len(out)), [o[1] for o in out], [o[2] for o in out]]
                            + [[o[3] == s for o in out] for s in ARXIV[1:]]).astype(float)
        keep = [True, True, True] + [X[:, c].sum() > 0 for c in range(3, X.shape[1])]
        b, se = logistic(X[:, keep], y)
        for c, lab in ((1, "egregious"), (2, "any annoying")):
            print(f"  logistic: {lab:12s} OR {math.exp(b[c]):.2f} (95% CI {math.exp(b[c] - 1.96 * se[c]):.2f}-"
                  f"{math.exp(b[c] + 1.96 * se[c]):.2f}), p = {math.erfc(abs(b[c] / se[c]) / math.sqrt(2)):.3g}")


if __name__ == "__main__":
    main()

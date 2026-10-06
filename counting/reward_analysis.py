#!/usr/bin/env python3
r"""
Analyse reward-model scores of the minimal pairs (reward_scores.py): the
reward difference original - deleted, i.e. what the "rather than Y" clause
adds to the score.

Per model: mean difference and share of items where the model prefers the
original, for ACL 2019 items and arXiv 2026 items, and within arXiv 2026 for
items each annotator labeled annoying vs. legitimate (Wilcoxon signed-rank
test against 0; Mann-Whitney between groups). A regression of the
difference on log clause length (in words) and the group indicators checks
that the clause effect is not just a length effect.

Usage:
    python3 reward_analysis.py ../data/reward_pairs/pairs.jsonl ../data/reward_pairs/scores.jsonl \
        ../data/annotation [--annotators A1 A2]
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def summary(label, diffs):
    if len(diffs) < 3:
        return f"  {label:34s} n={len(diffs)}"
    d = np.array(diffs)
    p = stats.wilcoxon(d).pvalue
    return (f"  {label:34s} n={len(d):4d}  mean {d.mean():+.3f}  median {np.median(d):+.3f}  "
            f"prefers original {np.mean(d > 0):5.1%}  Wilcoxon p = {p:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("scores")
    ap.add_argument("annotation_dir")
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    args = ap.parse_args()
    pairs = {p["item_id"]: p for p in map(json.loads, open(args.pairs))}
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(Path(args.annotation_dir) / "human_labels.jsonl")):
        labels[r["annotator"]][r["item_id"]] = r["label"]
    by_model = collections.defaultdict(dict)
    for r in map(json.loads, open(args.scores)):
        if r["item_id"] in pairs:
            by_model[r["model"]][r["item_id"]] = r["original"] - r["deleted"]
    for model, diff in by_model.items():
        acl = [diff[i] for i in diff if pairs[i]["corpus"] == "acl2019"]
        arx = [i for i in diff if pairs[i]["corpus"] != "acl2019"]
        print(f"\n== {model}")
        print(summary("ACL 2019", acl))
        print(summary("arXiv 2026", [diff[i] for i in arx]))
        if len(acl) > 2 and len(arx) > 2:
            print(f"  ACL 2019 vs arXiv 2026: Mann-Whitney p = {stats.mannwhitneyu(acl, [diff[i] for i in arx]).pvalue:.3g}")
        for name in args.annotators:
            ann = [diff[i] for i in arx if labels[name].get(i) == "annoying"]
            leg = [diff[i] for i in arx if labels[name].get(i) == "legitimate"]
            print(summary(f"arXiv, {name} annoying", ann))
            print(summary(f"arXiv, {name} legitimate", leg))
            if len(ann) > 2 and len(leg) > 2:
                print(f"  annoying vs legitimate ({name}): Mann-Whitney p = {stats.mannwhitneyu(ann, leg).pvalue:.3g}")
        ids = list(diff)
        y = np.array([diff[i] for i in ids])
        X = np.column_stack([np.ones(len(ids)),
                             [math.log(len(pairs[i]["clause"].split())) for i in ids],
                             [pairs[i]["corpus"] != "acl2019" for i in ids]]).astype(float)
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        sigma2 = ((y - X @ coef) ** 2).sum() / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
        for k, lab in ((0, "intercept"), (1, "log clause length"), (2, "arXiv 2026")):
            print(f"  OLS {lab:18s} {coef[k]:+.3f} (SE {se[k]:.3f}), p = {math.erfc(abs(coef[k] / se[k]) / math.sqrt(2)):.3g}")


if __name__ == "__main__":
    main()

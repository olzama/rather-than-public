#!/usr/bin/env python3
r"""
Does the reward gain from the "rather than Y" clause differ between clauses an
annotator found annoying and ones they found legitimate, once clause length is
held fixed? reward_analysis.py compares the two groups with a Mann-Whitney test
that ignores length; clause length is the strongest predictor of the reward gain
and annotators' groups can differ in length, so here, per model and annotator
(arXiv 2026 items only):

  (original - deleted) ~ intercept + b1 * annoying + b2 * log(clause words)

and the length of the annoying vs legitimate clauses is compared too. b1 is the
extra reward gain for annoying clauses, in reward points. Plain OLS, two-sided
t-test; no correction for the several models and annotators.

Usage:
    python3 reward_annotation_length.py ../data/reward_pairs/pairs.jsonl \
        ../data/reward_pairs/scores.jsonl ../data/annotation
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


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
    nwords = {i: max(1, len(p["clause"].split())) for i, p in pairs.items()}
    for name in args.annotators:
        ids = [i for i, p in pairs.items() if p["corpus"] != "acl2019"
               and labels[name].get(i) in ("annoying", "legitimate")]
        ann = np.array([labels[name][i] == "annoying" for i in ids], float)
        length = np.array([nwords[i] for i in ids], float)
        p_len = stats.mannwhitneyu(length[ann == 1], length[ann == 0]).pvalue
        print(f"\n== {name}: arXiv n={len(ids)}, annoying {int(ann.sum())}; clause words "
              f"annoying {length[ann == 1].mean():.1f} vs legitimate {length[ann == 0].mean():.1f} "
              f"(Mann-Whitney p = {p_len:.3g})")
        X = np.column_stack([np.ones(len(ids)), ann, np.log(length)])
        for model, diff in by_model.items():
            y = np.array([diff[i] for i in ids])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            df = len(y) - X.shape[1]
            se = np.sqrt(np.diag(resid @ resid / df * np.linalg.inv(X.T @ X)))
            p = 2 * stats.t.sf(abs(beta[1] / se[1]), df)
            print(f"  {model:40s} annoying {beta[1]:+.2f} (SE {se[1]:.2f}), p = {p:.3g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Length-matched control for the reward-model test (build_length_controls.py).
Per model, on the items that have a control, three texts are compared:
original (with "rather than Y"), control (Y clause replaced by a
non-contrastive clause of the same length) and deleted (clause removed).

  original - deleted   what the clause adds (antithesis + extra text)
  control  - deleted   what extra text of the same length adds
  original - control   the antithesis effect net of length (Wilcoxon signed-rank vs 0)

Also reported for ACL 2019 and arXiv 2026 separately, and the Spearman
correlation of the control effect with clause length (words).

Usage:
    python3 reward_control_analysis.py ../data/reward_pairs/scores.jsonl \
        ../data/reward_pairs/pairs_control.jsonl ../data/reward_pairs/scores_control.jsonl
"""
import collections
import json
import sys

import numpy as np
from scipy import stats


def line(label, d):
    d = np.array(d)
    return (f"  {label:22s} n={len(d):4d}  mean {d.mean():+.3f}  median {np.median(d):+.3f}  "
            f">0: {np.mean(d > 0):5.1%}  Wilcoxon p = {stats.wilcoxon(d).pvalue:.3g}")


def main():
    scores_path, ctl_pairs_path, ctl_scores_path = sys.argv[1:4]
    pairs = {p["item_id"]: p for p in map(json.loads, open(ctl_pairs_path))}
    orig = collections.defaultdict(dict)
    for r in map(json.loads, open(scores_path)):
        orig[r["model"]][r["item_id"]] = r
    ctl = collections.defaultdict(dict)
    for r in map(json.loads, open(ctl_scores_path)):
        ctl[r["model"]][r["item_id"]] = r
    for model in ctl:
        ids = [i for i in ctl[model] if i in orig[model] and i in pairs]
        if len(ids) < 3:
            continue
        o = np.array([orig[model][i]["original"] for i in ids])
        d = np.array([orig[model][i]["deleted"] for i in ids])
        c = np.array([ctl[model][i]["original"] for i in ids])
        print(f"\n== {model}  ({len(ids)} items with a control)")
        print(line("original - deleted", o - d))
        print(line("control - deleted", c - d))
        print(line("original - control", o - c))
        for name, is_acl in (("ACL 2019", True), ("arXiv 2026", False)):  # as in reward_analysis.py
            sel = [k for k, i in enumerate(ids) if (pairs[i]["corpus"] == "acl2019") == is_acl]
            if len(sel) >= 3:
                print(line(f"  {name}: orig - ctl", (o - c)[sel]))
        n_words = [len(pairs[i]["replacement"].split()) for i in ids]
        rho, p = stats.spearmanr(n_words, c - d)
        print(f"  Spearman (control - deleted) vs clause length: rho = {rho:+.2f}, p = {p:.3g}")


if __name__ == "__main__":
    main()

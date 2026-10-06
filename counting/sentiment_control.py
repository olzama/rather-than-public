#!/usr/bin/env python3
r"""
Sentiment control for the reward-model test. Takes LLM sentiment ratings
(-2..+2) of the original, control (build_length_controls.py) and deleted
version of every item (data_collection/sentiment_llm.py) and asks whether the
reward differences survive once the sentiment differences are held fixed:

  reward(a) - reward(b) ~ intercept + b1 * (sentiment(a) - sentiment(b)) [+ b2 * log clause length]

for original - deleted (with the length term) and original - control (same
length by construction), per reward model. The intercept is the clause effect
net of sentiment (and length); the sentiment slope says how much reward tracks
sentiment. Run after reward_scores.py on pairs.jsonl and pairs_control.jsonl.

Usage:
    python3 sentiment_control.py ../data/reward_pairs/pairs_control.jsonl ../data/reward_pairs/scores.jsonl \
        ../data/reward_pairs/scores_control.jsonl ../data/reward_pairs/sentiment.jsonl
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

CLASSIFIERS = {"llm": None}  # rating key in the sentiment file (data_collection/sentiment_llm.py)


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - X.shape[1]
    se = np.sqrt(np.diag(resid @ resid / df * np.linalg.inv(X.T @ X)))
    return beta, se, 2 * stats.t.sf(np.abs(beta / se), df)


def main():
    pairs_path, scores_path, ctl_scores_path, sent_path = sys.argv[1:5]
    pairs = [json.loads(l) for l in open(pairs_path)]
    if not Path(sent_path).exists():
        sys.exit(f"{sent_path} not found: run data_collection/sentiment_llm.py first")
    sent = collections.defaultdict(dict)
    for r in map(json.loads, open(sent_path)):
        sent[r["item_id"]][r["text"]] = r
    print("Mean LLM sentiment rating (-2..+2):")
    for c in CLASSIFIERS:
        print("  " + c + ": " + ", ".join(
            f"{k} {np.mean([sent[p['item_id']][k][c] for p in pairs]):+.3f}" for k in ("original", "control", "deleted")))
    orig, ctl = collections.defaultdict(dict), collections.defaultdict(dict)
    for r in map(json.loads, open(scores_path)):
        orig[r["model"]][r["item_id"]] = r
    for r in map(json.loads, open(ctl_scores_path)):
        ctl[r["model"]][r["item_id"]] = r
    for model in ctl:
        ids = [p for p in pairs if p["item_id"] in ctl[model] and p["item_id"] in orig[model]]
        if len(ids) < 10:
            continue
        o = np.array([orig[model][p["item_id"]]["original"] for p in ids])
        d = np.array([orig[model][p["item_id"]]["deleted"] for p in ids])
        c = np.array([ctl[model][p["item_id"]]["original"] for p in ids])
        clause = np.array([len(" ".join(p["clause"].split()).lstrip(" ,").split()) for p in ids])
        rep = np.array([len(p["replacement"].split()) for p in ids])
        print(f"\n== {model}  ({len(ids)} items)")
        for cname in CLASSIFIERS:
            s = {k: np.array([sent[p["item_id"]][k][cname] for p in ids]) for k in ("original", "control", "deleted")}
            for label, y, sa, sb, ln in (("original - deleted", o - d, s["original"], s["deleted"], np.log(clause)),
                                         ("original - control", o - c, s["original"], s["control"],
                                          np.log(clause) - np.log(rep))):
                dsent = sa - sb
                X = [dsent] if label == "original - control" else [dsent, ln]
                b, se, p = ols(y, np.column_stack(X))
                names = ["sentiment diff"] + ([] if label == "original - control" else ["log clause length"])
                rho = stats.spearmanr(dsent, y)
                print(f"  [{cname}] {label}: raw mean {y.mean():+.3f}; sentiment diff mean {dsent.mean():+.3f}; "
                      f"corr w/ reward diff rho = {rho.statistic:+.2f}")
                print(f"      OLS intercept {b[0]:+.3f} (SE {se[0]:.3f}, p = {p[0]:.3g}); " + "; ".join(
                    f"{n} {b[k + 1]:+.3f} (SE {se[k + 1]:.3f}, p = {p[k + 1]:.3g})" for k, n in enumerate(names)))


if __name__ == "__main__":
    main()

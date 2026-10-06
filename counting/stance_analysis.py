#!/usr/bin/env python3
r"""
Evaluative stance toward X and Y (stance_llm.py, or any scorer writing
item_id, x, y) against annoyance labels and straw-man codes. Specified
before the scores were produced:

  gap = x - y (how much more favourably X is presented than Y).
  H1  arXiv 2026 items: the gap is larger, and y lower, for items the
      annotator labeled annoying than for items labeled legitimate (each
      annotator; Either = annoying if either annotator labeled it so,
      legitimate if both did). One-sided Mann-Whitney tests; logistic
      regression of annoying on the gap with stratum fixed effects.
  H2  Straw-man coded items: y is lower for items a coder coded as straw
      man than for items coded not a straw man (each coder; one-sided
      Mann-Whitney).
  Robustness (added after the tests above): the same tests on non-inverted
  items only (x >= y). In an inverted item the author favours the rejected
  alternative Y ("prior work does X rather than Y", Y being the author's own
  approach); inverted items are almost all labeled legitimate.
  Descriptive: distribution of x, y and gap; ACL 2019 vs. arXiv 2026 pool
  items. With --compare, agreement with a second scorer (Spearman on x, y
  and gap).

Usage:
    python3 stance_analysis.py ../data/annotation ../data/annotation/stance/stance__gpt-6-sol.jsonl \
        [--compare other_scores.jsonl] [--annotators A1 A2] [--batch 1]
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ARXIV = ["arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"]
CORPORA = list(ARXIV)  # corpora compared in H1 (--corpora)


def logistic(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
        b += np.linalg.solve(H, X.T @ (y - p))
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def h1(sc, prov, labels, names, subset, keep):
    print(f"\nH1: annoying vs legitimate ({', '.join(CORPORA)}), {subset}")
    for name in names:
        ids = [i for i in sc if prov[i]["corpus"] in CORPORA and keep(i)
               and labels[name].get(i) in ("annoying", "legitimate")]
        ann = [i for i in ids if labels[name][i] == "annoying"]
        leg = [i for i in ids if labels[name][i] == "legitimate"]
        if len(ann) < 3 or len(leg) < 3:
            continue
        gap = lambda L: [sc[i][0] - sc[i][1] for i in L]
        yv = lambda L: [sc[i][1] for i in L]
        pg = stats.mannwhitneyu(gap(ann), gap(leg), alternative="greater").pvalue
        py = stats.mannwhitneyu(yv(ann), yv(leg), alternative="less").pvalue
        X = np.column_stack([np.ones(len(ids)), gap(ids)] + [[prov[i]["corpus"] == s for i in ids] for s in CORPORA[1:]]).astype(float)
        cols = [True, True] + [X[:, k].sum() > 0 for k in range(2, X.shape[1])]
        coef, se = logistic(X[:, cols], np.array([labels[name][i] == "annoying" for i in ids], float))
        print(f"  {name:7s} annoying n={len(ann)}, legitimate n={len(leg)}: gap {np.mean(gap(ann)):+.2f} vs {np.mean(gap(leg)):+.2f} (p = {pg:.3g}); "
              f"y {np.mean(yv(ann)):+.2f} vs {np.mean(yv(leg)):+.2f} (p = {py:.3g}); "
              f"logistic OR per gap point {math.exp(coef[1]):.2f} ({math.exp(coef[1] - 1.96 * se[1]):.2f}-{math.exp(coef[1] + 1.96 * se[1]):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("scores")
    ap.add_argument("--compare")
    ap.add_argument("--annotators", nargs=2, default=["A1", "A2"])
    ap.add_argument("--batch", type=int)
    ap.add_argument("--corpora", nargs="+", default=ARXIV, help="corpora compared in H1")
    args = ap.parse_args()
    CORPORA[:] = args.corpora
    d = Path(args.annotation_dir)
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl"))
            if "repeat_of" not in r and "excluded" not in r}
    pub = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl"))}
    sc = {r["item_id"]: (r["x"], r["y"]) for r in map(json.loads, open(args.scores))
          if r["item_id"] in prov and r["item_id"] in pub
          and (args.batch is None or pub[r["item_id"]].get("batch", 1) == args.batch)}
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        labels[r["annotator"]][r["item_id"]] = r["label"]
    codes = collections.defaultdict(dict)
    if (d / "strawman" / "human_codes.jsonl").exists():
        for r in map(json.loads, open(d / "strawman" / "human_codes.jsonl")):
            codes[r["annotator"]][r["item_id"]] = r["code"]
    a, b = args.annotators
    labels["Either"] = {i: ("annoying" if "annoying" in (labels[a][i], labels[b][i]) else
                            "legitimate" if labels[a][i] == labels[b][i] == "legitimate" else "other")
                        for i in labels[a] if i in labels[b]}

    xs = [v[0] for v in sc.values()]
    ys = [v[1] for v in sc.values()]
    print(f"{len(sc)} scored items; x mean {np.mean(xs):+.2f}, y mean {np.mean(ys):+.2f}, gap mean {np.mean(np.subtract(xs, ys)):+.2f}")
    print("  y distribution (rounded):", dict(sorted(collections.Counter(int(round(v)) for v in ys).items())))
    print("  x distribution (rounded):", dict(sorted(collections.Counter(int(round(v)) for v in xs).items())))
    for grp, test in (("ACL 2019", lambda c: c == "acl2019"), ("arXiv 2026", lambda c: c != "acl2019")):
        g = [sc[i][0] - sc[i][1] for i in sc if test(prov[i]["corpus"])]
        print(f"  {grp}: n={len(g)}, gap mean {np.mean(g):+.2f}, share with y<0 "
              f"{np.mean([sc[i][1] < 0 for i in sc if test(prov[i]['corpus'])]):.1%}")

    for subset, keep in (("all items", lambda i: True),
                         ("non-inverted items (x >= y)", lambda i: sc[i][0] >= sc[i][1])):
        h1(sc, prov, labels, (a, b, "Either"), subset, keep)

    print("\nInverted items (x < y: the author favours the rejected alternative), arXiv 2026")
    for name in (a, b, "Either"):
        ids = [i for i in sc if prov[i]["corpus"] in ARXIV and labels[name].get(i) in ("annoying", "legitimate")]
        for lab in ("annoying", "legitimate"):
            g = [i for i in ids if labels[name][i] == lab]
            print(f"  {name:7s} {lab:10s}: {sum(sc[i][0] < sc[i][1] for i in g)}/{len(g)}")

    print("\nH2: straw man vs not (coded items)")
    for coder in sorted(codes):
        for subset, keep in (("all", lambda i: True), ("non-inverted", lambda i: sc[i][0] >= sc[i][1])):
            sm = [sc[i][1] for i in codes[coder] if i in sc and keep(i) and codes[coder][i] == "strawman"]
            ns = [sc[i][1] for i in codes[coder] if i in sc and keep(i) and codes[coder][i] == "not_strawman"]
            if len(sm) >= 3 and len(ns) >= 3:
                p = stats.mannwhitneyu(sm, ns, alternative="less").pvalue
                print(f"  {coder:7s} {subset:12s}: y straw man {np.mean(sm):+.2f} (n={len(sm)}) vs not {np.mean(ns):+.2f} (n={len(ns)}), p = {p:.3g}")

    if args.compare:
        other = {r["item_id"]: (r["x"], r["y"]) for r in map(json.loads, open(args.compare))}
        both = [i for i in sc if i in other]
        if len(both) > 10:
            print(f"\nagreement with {Path(args.compare).name} (n={len(both)}):")
            for k, lab in ((0, "x"), (1, "y")):
                rho, p = stats.spearmanr([sc[i][k] for i in both], [other[i][k] for i in both])
                print(f"  {lab}: Spearman {rho:+.2f}, p = {p:.3g}")
            rho, p = stats.spearmanr([sc[i][0] - sc[i][1] for i in both], [other[i][0] - other[i][1] for i in both])
            print(f"  gap: Spearman {rho:+.2f}, p = {p:.3g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Do an annotator's criteria shift between labeling periods? The pool was
labeled in two batches, the second after the first had been analyzed. For
each annotator, the label-based results are recomputed within each batch,
and the change between batches is tested:

  rates      annoying rate of ACL 2019 and of random arXiv 2026 items;
  stance     annoying vs legitimate arXiv 2026 items: mean gap X-Y and mean
             Y (GPT-6-sol ratings); the batch difference is the interaction
             term of a logistic regression of annoying on gap, batch and
             gap x batch;
  syntax     preselected features (syntax_features.py --out files); the
             paper's per-batch syntax comes from syntax_patterns.py --batch;
  repeats    hidden repeats: labels that change into or out of annoying.

Usage:
    python3 drift_by_batch.py ../data/annotation --syntax-dir DIR [--annotators A1 A2]
(DIR holds syn_<annotator>.jsonl from syntax_features.py --annotator NAME --out ...)
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ARXIV = {"arxiv2026", "arxiv_v1", "arxiv_latest"}
FEATURES = [("initial", None), ("passive", None), ("cat_X", "AdjP"), ("cat_Y", "AdjP"), ("cat_Y", "VP"), ("len_Y", "median")]


def logistic(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
        b += np.linalg.solve(H, X.T @ (y - p))
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--syntax-dir", required=True)
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl"))
            if "repeat_of" not in r and not r.get("excluded")}
    batch = {r["item_id"]: r.get("batch", 1) for r in map(json.loads, open(d / "items_public.jsonl"))}
    sc = {r["item_id"]: (r["x"], r["y"]) for r in map(json.loads, open(d / "stance" / "stance__gpt-6-sol.jsonl"))}
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        labels[r["annotator"]][r["item_id"]] = r["label"]

    for name in args.annotators:
        L = labels[name]
        print(f"\n===== {name}")
        print("rates (annoying / annoying+legitimate+unsure, garbled excluded):")
        for b in (1, 2):
            for grp, test in (("ACL 2019", lambda c: c == "acl2019"), ("arXiv 2026", lambda c: c in ARXIV)):
                ids = [i for i in prov if batch.get(i) == b and test(prov[i]["corpus"]) and L.get(i) not in (None, "garbled")]
                k = sum(L[i] == "annoying" for i in ids)
                print(f"  batch {b} {grp:10s} {k}/{len(ids)} ({k / len(ids):.1%})")
        for grp, test in (("arXiv 2026", lambda c: c in ARXIV),):
            t = [[sum(L.get(i) == "annoying" for i in prov if batch.get(i) == b and test(prov[i]["corpus"]) and L.get(i) not in (None, "garbled")),
                  sum(L.get(i) not in (None, "garbled", "annoying") for i in prov if batch.get(i) == b and test(prov[i]["corpus"]))] for b in (1, 2)]
            print(f"  arXiv rate batch 1 vs 2: Fisher p = {stats.fisher_exact(t)[1]:.3g}")

        print("stance, annoying vs legitimate arXiv items (gap X-Y; Y):")
        rows = []
        for b in (1, 2):
            ids = [i for i in prov if batch.get(i) == b and prov[i]["corpus"] in ARXIV and i in sc and L.get(i) in ("annoying", "legitimate")]
            ann = [i for i in ids if L[i] == "annoying"]
            leg = [i for i in ids if L[i] == "legitimate"]
            g = lambda S: [sc[i][0] - sc[i][1] for i in S]
            y = lambda S: [sc[i][1] for i in S]
            pg = stats.mannwhitneyu(g(ann), g(leg), alternative="greater").pvalue
            py = stats.mannwhitneyu(y(ann), y(leg), alternative="less").pvalue
            print(f"  batch {b}: n={len(ann)}/{len(leg)}  gap {np.mean(g(ann)):+.2f} vs {np.mean(g(leg)):+.2f} (p = {pg:.3g});"
                  f"  Y {np.mean(y(ann)):+.2f} vs {np.mean(y(leg)):+.2f} (p = {py:.3g})")
            rows += [(sc[i][0] - sc[i][1], b == 2, L[i] == "annoying") for i in ids]
        gap = np.array([r[0] for r in rows]); b2 = np.array([r[1] for r in rows], float); yv = np.array([r[2] for r in rows], float)
        X = np.column_stack([np.ones(len(rows)), gap, b2, gap * b2])
        coef, se = logistic(X, yv)
        z = coef[3] / se[3]
        print(f"  gap x batch-2 interaction: OR {math.exp(coef[3]):.2f} per point (p = {math.erfc(abs(z) / math.sqrt(2)):.3g})"
              f"  [gap OR in batch 1 {math.exp(coef[1]):.2f}, batch 2 {math.exp(coef[1] + coef[3]):.2f}]")

        syn = {r["item_id"]: r for r in map(json.loads, open(Path(args.syntax_dir) / f"syn_{name}.jsonl"))}
        print("syntax, annoying / legitimate arXiv items (% or median):")
        for feat, val in FEATURES:
            out = []
            for b in (1, 2):
                S = [r for i, r in syn.items() if batch.get(i) == b]
                ann = [r for r in S if r["label"] == "annoying"]
                leg = [r for r in S if r["label"] == "legitimate"]
                if val == "median":
                    p = stats.mannwhitneyu([r[feat] for r in ann], [r[feat] for r in leg]).pvalue
                    out.append(f"b{b} {np.median([r[feat] for r in ann]):g}/{np.median([r[feat] for r in leg]):g} (p={p:.2g})")
                else:
                    f = (lambda r: bool(r[feat])) if val is None else (lambda r: r[feat] == val)
                    a, c = sum(map(f, ann)), sum(map(f, leg))
                    p = stats.fisher_exact([[a, len(ann) - a], [c, len(leg) - c]])[1]
                    out.append(f"b{b} {100 * a / len(ann):.0f}%/{100 * c / len(leg):.0f}% (p={p:.2g})")
            print(f"  {feat + ('=' + val if val and val != 'median' else ''):12s} " + "   ".join(out))

    print("\nhidden repeats (first label -> repeat label), by annotator:")
    for name in args.annotators:
        L = labels[name]
        ch = collections.Counter()
        pairs = {i: i[5:] for i in L if i.startswith("rep__")}  # later repeats: rep__<item>
        pairs.update({r["item_id"]: r["repeat_of"] for r in map(json.loads, open(d / "items_provenance.jsonl"))
                      if "repeat_of" in r and r["item_id"] in L})  # first repeats: separate items with repeat_of
        for rep, orig in pairs.items():
            if orig in L:
                ch[(L[orig], L[rep])] += 1
        into = sum(v for (a, b), v in ch.items() if b == "annoying" and a != "annoying")
        out = sum(v for (a, b), v in ch.items() if a == "annoying" and b != "annoying")
        same = sum(v for (a, b), v in ch.items() if a == b)
        print(f"  {name}: {sum(ch.values())} repeats, same {same}; into annoying {into}, out of annoying {out}")


if __name__ == "__main__":
    main()

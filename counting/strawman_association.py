#!/usr/bin/env python3
r"""
Straw-man codes against annoyance labels (the paper's straw-man table).

For each coder x annotator pair: share of test items coded straw man among
the items the annotator labeled annoying and among those labeled legitimate,
with Fisher's exact test (two-sided). Items coded or labeled unsure, and
garbled items, are omitted. Also Cohen's kappa between coders (straw man vs.
other, all coded items both coders coded). Also, per annotator, the same
comparison using only the other coders' codes (independent of that annotator).

Usage:
    python3 strawman_association.py ../data/annotation \
        [--coders A1 A2 A3 A4] [--annotators A1 A2]
"""
import argparse
import collections
import itertools
import json
from pathlib import Path

from scipy import stats


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--coders", nargs="+", default=["A1", "A2", "A3", "A4"])
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        labels[r["annotator"]][r["item_id"]] = r["label"]
    codes = collections.defaultdict(dict)
    sets = {}
    for r in map(json.loads, open(d / "strawman" / "human_codes.jsonl")):
        codes[r["annotator"]][r["item_id"]] = r["code"]
        sets[r["item_id"]] = r["set"]

    for c in args.coders:
        cc = collections.Counter((sets[i], v) for i, v in codes[c].items())
        print(f"{c}: {len(codes[c])} coded; " + ", ".join(f"{k[0]}/{k[1]} {n}" for k, n in sorted(cc.items())))

    print("\ncodes / labels: straw man among annoying vs legitimate (test set)")
    for c, a in itertools.product(args.coders, args.annotators):
        cell = {}
        for lab in ("annoying", "legitimate"):
            ids = [i for i, v in codes[c].items() if sets[i] == "test" and v in ("strawman", "not_strawman")
                   and labels[a].get(i) == lab]
            cell[lab] = (sum(codes[c][i] == "strawman" for i in ids), len(ids))
        (sa, na), (sl, nl) = cell["annoying"], cell["legitimate"]
        if na and nl:
            p = stats.fisher_exact([[sa, na - sa], [sl, nl - sl]]).pvalue
            print(f"  {c:7s} / {a:7s}: {sa}/{na} ({sa / na:.0%}) vs {sl}/{nl} ({sl / nl:.0%}), p = {p:.3g}")

    print("\nindependent coders: for each annotator's labels, the share of the OTHER coders")
    print("(not the annotator) who code an item a straw man, averaged per item; items coded")
    print("straw man / not by all of those coders; one-sided Mann-Whitney (annoying > legitimate)")
    for a in args.annotators:
        others = [c for c in args.coders if c != a]
        ids = [i for i in codes[others[0]] if sets[i] == "test" and labels[a].get(i) in ("annoying", "legitimate")
               and all(codes[c].get(i) in ("strawman", "not_strawman") for c in others)]
        share = {i: sum(codes[c][i] == "strawman" for c in others) / len(others) for i in ids}
        ann = [share[i] for i in ids if labels[a][i] == "annoying"]
        leg = [share[i] for i in ids if labels[a][i] == "legitimate"]
        if ann and leg:
            p = stats.mannwhitneyu(ann, leg, alternative="greater").pvalue
            print(f"  labels {a:7s}, coders {'+'.join(others)}: {sum(ann) / len(ann):.2f} (n={len(ann)}) vs "
                  f"{sum(leg) / len(leg):.2f} (n={len(leg)}), p = {p:.3g}")

    print("\ncoders who never judged annoyance (not an annotator), same test")
    blind = [c for c in args.coders if c not in args.annotators]
    for a in args.annotators:
        ids = [i for i in codes[blind[0]] if sets[i] == "test" and labels[a].get(i) in ("annoying", "legitimate")
               and all(codes[c].get(i) in ("strawman", "not_strawman") for c in blind)]
        share = {i: sum(codes[c][i] == "strawman" for c in blind) / len(blind) for i in ids}
        ann = [share[i] for i in ids if labels[a][i] == "annoying"]
        leg = [share[i] for i in ids if labels[a][i] == "legitimate"]
        if ann and leg:
            p = stats.mannwhitneyu(ann, leg, alternative="greater").pvalue
            print(f"  labels {a:7s}, coders {'+'.join(blind)}: {sum(ann) / len(ann):.2f} (n={len(ann)}) vs "
                  f"{sum(leg) / len(leg):.2f} (n={len(leg)}), p = {p:.3g}")

    print("\nkappa, straw man vs other (items both coded)")
    for c1, c2 in itertools.combinations(args.coders, 2):
        both = sorted(set(codes[c1]) & set(codes[c2]))
        if len(both) > 5:
            k = kappa([codes[c1][i] == "strawman" for i in both], [codes[c2][i] == "strawman" for i in both])
            s1 = {i for i in both if codes[c1][i] == "strawman"}
            s2 = {i for i in both if codes[c2][i] == "strawman"}
            print(f"  {c1}-{c2}: n={len(both)}, kappa {k:.2f}; straw men {len(s1)} / {len(s2)}, shared {len(s1 & s2)}")


if __name__ == "__main__":
    main()

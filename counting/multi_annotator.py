#!/usr/bin/env python3
r"""
Compare annotators on the main annotation pool (hidden repeats excluded).

1. Per annotator: annoying rate per corpus group (ACL 2019; arXiv 2026
   random = dataset (2) + v1 + latest; high-count arXiv 2026), items
   the annotator marked garbled excluded, and Fisher's exact test of ACL
   2019 vs arXiv 2026 random.
2. Agreement between two annotators on the items both labeled and
   neither marked garbled: Cohen's kappa (3-way and annoying vs. not),
   Gwet's AC1 (annoying vs. not; less sensitive to a rare category),
   positive agreement on "annoying" (2 * both / (A + B)), and the
   confusion table.
3. Band: per group, the rate of items both annotators call annoying
   (lower bound) and either calls annoying (upper bound), with Fisher's
   test of ACL 2019 vs arXiv 2026 random for each.

Usage:
    python3 multi_annotator.py ../data/annotation --pair A1 A2 [--also A5]
"""
import argparse
import collections
import json
import math
from pathlib import Path

from scipy import stats

GROUPS = {"acl2019": "ACL 2019", "arxiv2026": "arXiv 2026 random", "arxiv_v1": "arXiv 2026 random",
          "arxiv_latest": "arXiv 2026 random", "high_count_arxiv2026": "High-count 2026",
          "llm_gpt-4o": "GPT-4o", "llm_gpt-5.6-sol": "GPT-5.6-sol, 6-sol", "llm_gpt-6-sol": "GPT-5.6-sol, 6-sol"}
ORDER = ["ACL 2019", "arXiv 2026 random", "High-count 2026", "GPT-4o", "GPT-5.6-sol, 6-sol"]
LABELS = ["annoying", "legitimate", "unsure"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n:.1%}; {lo:.1%}-{hi:.1%})" if n else "-"


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = collections.Counter(x), collections.Counter(y)
    pe = sum(cx[v] * cy[v] for v in set(x) | set(y)) / (n * n)
    return (po - pe) / (1 - pe)


def ac1(x, y):
    """Gwet's AC1 for two raters and binary labels."""
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    pi = (sum(x) + sum(y)) / (2 * n)
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe)


def acl_test(k_acl, n_acl, k_ax, n_ax):
    return stats.fisher_exact([[k_acl, n_acl - k_acl], [k_ax, n_ax - k_ax]])[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--pair", nargs=2, default=["A1", "A2"])
    ap.add_argument("--batch", type=int, help="only items of this annotation batch (default: all)")
    ap.add_argument("--also", nargs="*", default=[])
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl")) if "excluded" not in r}
    pool = {r["item_id"] for r in map(json.loads, open(d / "items_public.jsonl"))
            if args.batch is None or r.get("batch", 1) == args.batch}
    pool &= set(prov)
    labels = collections.defaultdict(dict)
    for l in open(d / "human_labels.jsonl"):
        r = json.loads(l)
        if r["item_id"] in pool:
            labels[r["annotator"]][r["item_id"]] = r["label"]

    print("1. Annoying rate per annotator (own garbled items excluded)\n")
    for name in args.pair + args.also:
        lab = labels[name]
        by = collections.defaultdict(lambda: [0, 0])
        for i, v in lab.items():
            if v == "garbled":
                continue
            g = GROUPS[prov[i]["corpus"]]
            by[g][0] += v == "annoying"
            by[g][1] += 1
        p = acl_test(*by["ACL 2019"], *by["arXiv 2026 random"])
        print(f"{name} ({len(lab)}/{len(pool)} labeled)")
        for g in ORDER:
            print(f"  {g:18s} {rate(*by[g])}")
        print(f"  ACL 2019 vs arXiv 2026 random: Fisher p = {p:.3g}\n")

    a, b = args.pair
    common = sorted(i for i in labels[a] if i in labels[b] and "garbled" not in (labels[a][i], labels[b][i]))
    xa, xb = [labels[a][i] for i in common], [labels[b][i] for i in common]
    ba, bb = [v == "annoying" for v in xa], [v == "annoying" for v in xb]
    both = sum(p and q for p, q in zip(ba, bb))
    print(f"2. Agreement, {a} vs {b}: {len(common)} items (garbled by either excluded)\n")
    print(f"  raw agreement (3-way)        {sum(p == q for p, q in zip(xa, xb)) / len(common):.1%}")
    print(f"  Cohen's kappa (3-way)        {kappa(xa, xb):.3f}")
    print(f"  Cohen's kappa (annoying)     {kappa(ba, bb):.3f}")
    print(f"  Gwet's AC1 (annoying)        {ac1(ba, bb):.3f}")
    print(f"  positive agreement (annoying) {2 * both / (sum(ba) + sum(bb)):.3f}  ({both} both; {sum(ba)} {a}, {sum(bb)} {b})")
    ct = collections.Counter(zip(xa, xb))
    print(f"\n  {a + ' \\ ' + b:22s}" + "".join(f"{v:>12s}" for v in LABELS))
    for u in LABELS:
        print(f"  {u:22s}" + "".join(f"{ct[(u, v)]:12d}" for v in LABELS))

    print(f"\n3. Band on the common items: both annoying (lower) / either annoying (upper)\n")
    by = collections.defaultdict(lambda: [0, 0, 0])
    for i, p, q in zip(common, ba, bb):
        g = GROUPS[prov[i]["corpus"]]
        by[g][0] += p and q
        by[g][1] += p or q
        by[g][2] += 1
    for g in ORDER:
        k_both, k_either, n = by[g]
        print(f"  {g:18s} both {rate(k_both, n):32s} either {rate(k_either, n)}")
    for j, name in ((0, "both"), (1, "either")):
        p = acl_test(by["ACL 2019"][j], by["ACL 2019"][2], by["arXiv 2026 random"][j], by["arXiv 2026 random"][2])
        print(f"  ACL 2019 vs arXiv 2026 random ({name}): Fisher p = {p:.3g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Summarize the human annotation of the "rather than" pool:

1. Label counts and the annoying rate per stratum, with Wilson 95%
   intervals. Garbled items are excluded from the rate denominator
   (they are an extraction judgment, not a judgment of the construction);
   the 40 hidden repeat items are excluded throughout.
2. ACL 2019 vs. arXiv 2026 (random strata) comparison, Fisher's exact test.
3. Drift: agreement between each hidden repeat and the same annotator's
   label on its original. Repeats are the fixed copies in repeats.jsonl
   and the per-annotator copies the page saves as "rep__<item_id>".
4. Order effect: annoying rate by labeling position (quintiles), and a
   logistic regression of annoying on labeling position with stratum
   fixed effects.

Each annotator is reported separately.

Usage:
    python3 annotation_report.py ../data/annotation [--annotator NAME]
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

STRATA = ["acl2019", "arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def logistic(X, y, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        b += np.linalg.solve(H, X.T @ (y - p))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se


def report(annotator, prov, labels, repeat_of):
    print(f"==== annotator: {annotator} ====")
    items = [i for i in prov if i not in repeat_of]
    missing = [i for i in items if i not in labels]
    print(f"pool items (excluding repeats): {len(items)}; unlabeled: {len(missing)}\n")
    if len(missing) == len(items):
        return

    # 1. per-stratum rates
    print(f"{'stratum':22s} {'n':>4s} {'legit':>6s} {'annoy':>6s} {'unsure':>6s} {'garbl':>6s}   annoying rate (95% CI, garbled excluded)")
    by = collections.defaultdict(collections.Counter)
    for i in items:
        if i in labels:
            by[prov[i]["corpus"]][labels[i]["label"]] += 1
    for s in STRATA + ["total"]:
        c = sum(by.values(), collections.Counter()) if s == "total" else by[s]
        n = sum(c.values())
        judged = n - c["garbled"]
        if not judged:
            continue
        lo, hi = wilson(c["annoying"], judged)
        print(f"{s:22s} {n:4d} {c['legitimate']:6d} {c['annoying']:6d} {c['unsure']:6d} {c['garbled']:6d}   "
              f"{c['annoying'] / judged:6.1%} ({lo:.1%}-{hi:.1%})")

    # 2. 2019 vs 2026
    a = by["acl2019"]; b = by["arxiv2026"]
    table = [[a["annoying"], a["legitimate"] + a["unsure"]], [b["annoying"], b["legitimate"] + b["unsure"]]]
    odds, p = stats.fisher_exact(table)
    print(f"\nACL 2019 vs arXiv 2026 (random strata), annoying vs not: {table}, Fisher p = {p:.3g}")

    # 3. drift from hidden repeats: the annotator's label on each copy vs. on its original
    pairs = [(labels[o]["label"], labels[c]["label"]) for c, o in repeat_of.items() if c in labels and o in labels]
    if pairs:
        agree = sum(o == n for o, n in pairs)
        print(f"\nhidden repeats: {len(pairs)} re-labeled; same label {agree}/{len(pairs)} ({agree / len(pairs):.0%})")
        print("  original -> repeat:", dict(collections.Counter(f"{o}->{n}" for o, n in pairs if o != n)))
        b01 = sum(o != "annoying" and n == "annoying" for o, n in pairs)
        b10 = sum(o == "annoying" and n != "annoying" for o, n in pairs)
        print(f"  annoying: {sum(o == 'annoying' for o, _ in pairs)} originally, {sum(n == 'annoying' for _, n in pairs)} on repeat")
        if b01 + b10:
            pm = stats.binomtest(b01, b01 + b10).pvalue
            print(f"  became annoying {b01}, stopped being annoying {b10}; exact McNemar p = {pm:.3g}")

    # 4. order effect
    rows = sorted((labels[i]["ts"], i) for i in items
                  if i in labels and labels[i]["label"] != "garbled" and labels[i].get("ts") is not None)
    n = len(rows)
    if n < 50:
        print("\norder effect: skipped (fewer than 50 timestamped labels)")
        return
    print(f"\nannoying rate by labeling-order quintile (n={n}, garbled excluded):")
    for q in range(5):
        chunk = rows[q * n // 5:(q + 1) * n // 5]
        k = sum(labels[i]["label"] == "annoying" for _, i in chunk)
        print(f"  Q{q + 1}: {k}/{len(chunk)} = {k / len(chunk):.1%}")
    X, y = [], []
    for pos, (_, i) in enumerate(rows):
        s = prov[i]["corpus"]
        if s == "acl2019":
            continue  # few or no annoying labels in this stratum would make its coefficient diverge
        X.append([1.0, pos / (n - 1)] + [1.0 if s == t else 0.0 for t in STRATA[2:]])
        y.append(1.0 if labels[i]["label"] == "annoying" else 0.0)
    if 0 < sum(y) < len(y):
        bcoef, se = logistic(np.array(X), np.array(y))
        z = bcoef[1] / se[1]
        print(f"  logistic (non-ACL strata, stratum fixed effects): position coef {bcoef[1]:.2f} "
              f"(odds ratio first->last {math.exp(bcoef[1]):.2f}), SE {se[1]:.2f}, p = {math.erfc(abs(z) / math.sqrt(2)):.3g}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--annotator", help="report only this annotator (default: each annotator in turn)")
    args = ap.parse_args()
    d = Path(args.annotation_dir)

    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl")) if "excluded" not in r}
    by_annotator = collections.defaultdict(dict)
    for l in open(d / "human_labels.jsonl"):
        r = json.loads(l)
        by_annotator[r["annotator"]][r["item_id"]] = r

    # Hidden repeats: the fixed copies listed in repeats.jsonl, plus the
    # per-annotator copies the page stores as "rep__<item_id>".
    fixed = {}
    if (d / "repeats.jsonl").exists():
        fixed = {json.loads(l)["item_id"]: json.loads(l)["repeat_of"] for l in open(d / "repeats.jsonl")}

    names = [args.annotator] if args.annotator else sorted(by_annotator)
    for name in names:
        labels = by_annotator[name]
        repeat_of = dict(fixed)
        repeat_of.update({i: i[len("rep__"):] for i in labels if i.startswith("rep__")})
        report(name, prov, labels, repeat_of)


if __name__ == "__main__":
    main()

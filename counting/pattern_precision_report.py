#!/usr/bin/env python3
r"""
Precision of each antithesis pattern per corpus (judgments from
data_collection/pattern_precision.py), with Wilson 95% intervals, and the
2019-to-2026 rate ratio before and after correcting each rate for precision
(rate x precision; rates from count_antithesis_patterns.py --json).

Usage:
    python3 pattern_precision_report.py ../data/pattern_precision/judgments.jsonl \
        --rates acl2019=acl2019_rates.json arxiv2026=arxiv2026_rates.json
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import PATTERNS


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judgments")
    ap.add_argument("--rates", nargs="+", default=[], help="corpus=rates.json")
    args = ap.parse_args()
    cells = collections.defaultdict(lambda: [0, 0])
    for r in map(json.loads, open(args.judgments)):
        c = cells[(r["pattern_id"], r["corpus"])]
        c[0] += r["instance"] == "yes"
        c[1] += 1
    rates = {}
    for spec in args.rates:
        corpus, path = spec.split("=", 1)
        rates[corpus] = {pid: v["mean_per_1000w"] for pid, v in json.load(open(path))["patterns"].items()}
    print(f"{'pattern':26s} {'ACL 2019 precision':>24s} {'arXiv 2026 precision':>24s} {'ratio raw':>10s} {'corrected':>10s}")
    for pid, tier, label, _, _ in PATTERNS:
        out = []
        prec = {}
        for corpus in ("acl2019", "arxiv2026"):
            k, n = cells.get((pid, corpus), (0, 0))
            p, lo, hi = wilson(k, n)
            prec[corpus] = p
            out.append(f"{k:3d}/{n:<3d} {100 * p:5.0f}% ({100 * lo:3.0f}-{100 * hi:3.0f})" if n else f"{'--':>24s}")
        ratio = corr = ""
        if rates and all(pid in rates.get(c, {}) for c in ("acl2019", "arxiv2026")) and rates["acl2019"][pid] > 0:
            raw = rates["arxiv2026"][pid] / rates["acl2019"][pid]
            ratio = f"{raw:10.2f}"
            if all(prec.get(c) and not math.isnan(prec[c]) for c in prec) and prec["acl2019"] > 0:
                corr = f"{raw * prec['arxiv2026'] / prec['acl2019']:10.2f}"
        print(f"{label:26s} {out[0]:>24s} {out[1]:>24s} {ratio:>10s} {corr:>10s}")


if __name__ == "__main__":
    main()

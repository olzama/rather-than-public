#!/usr/bin/env python3
r"""
Paired test of whether revision changes the "rather than" rate: for each
paper in the version-diff subset, compare its v1 with its latest version
(highest vN), on body text (references and appendices removed).

Reports the mean per-paper rate per 1,000 words for both versions (the
figures in the paper's antithesis-family table), the mean paired
difference with a bootstrap 95% interval, the Wilcoxon signed-rank test
on per-paper differences, how many papers went up, down, or stayed the
same, and the same comparison for raw counts and body length.

Usage:
    python3 version_rate_test.py <arxiv_versions/text_body> [--pattern rather_than] [--boot 10000] \
        [--exclude ../data/excluded_documents.tsv]

--exclude drops a paper when either of its two versions is listed in
language_filter.py's output (corpus arxiv_versions).
"""
import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import compile_patterns  # noqa: E402

WORD_RE = re.compile(r"\w+")
VERSION_RE = re.compile(r"^(.*)_v(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_dir")
    ap.add_argument("--pattern", default="rather_than")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exclude", help="TSV from language_filter.py")
    args = ap.parse_args()

    rx = compile_patterns()[args.pattern]

    versions = defaultdict(dict)
    for f in Path(args.text_dir).glob("*_v*.txt"):
        m = VERSION_RE.match(f.stem)
        if m:
            versions[m.group(1)][int(m.group(2))] = f

    skip = set()
    if args.exclude:
        skip = {l.split("\t")[1] for l in list(open(args.exclude))[1:] if l.split("\t")[0] == "arxiv_versions"}

    rows, n_skipped = [], 0
    for paper, vs in versions.items():
        if 1 not in vs or len(vs) < 2:
            continue
        if {vs[1].stem, vs[max(vs)].stem} & skip:
            n_skipped += 1
            continue
        pair = []
        for v in (1, max(vs)):
            text = vs[v].read_text(errors="replace")
            pair.append((len(rx.findall(text)), len(WORD_RE.findall(text))))
        (c1, w1), (c2, w2) = pair
        if w1 and w2:
            rows.append((1000 * c1 / w1, 1000 * c2 / w2, c1, c2, w1, w2))
    r1, r2, c1, c2, w1, w2 = (np.array(x, dtype=float) for x in zip(*rows))

    diff = r2 - r1
    rng = np.random.default_rng(args.seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(args.boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    w = stats.wilcoxon(r2, r1)
    up, down = int((diff > 0).sum()), int((diff < 0).sum())
    print(f"pattern: {args.pattern}; paper pairs: {len(rows)} ({n_skipped} excluded)")
    print(f"mean rate per 1k words: v1 {r1.mean():.3f}, latest {r2.mean():.3f} "
          f"(ratio {r2.mean() / r1.mean():.3f})")
    print(f"mean paired difference {diff.mean():+.3f} per 1k (95% bootstrap CI {lo:+.3f} to {hi:+.3f}); "
          f"Wilcoxon signed-rank p = {w.pvalue:.3g}")
    print(f"papers with a higher rate in the latest version: {up}; lower: {down}; unchanged: {len(rows) - up - down}")
    print(f"sign test p = {stats.binomtest(up, up + down).pvalue:.3g}")
    print(f"raw count: v1 total {int(c1.sum())}, latest total {int(c2.sum())}; Wilcoxon p = {stats.wilcoxon(c2, c1).pvalue:.3g}")
    print(f"body length: v1 median {statistics.median(w1):.0f} words, latest {statistics.median(w2):.0f}; "
          f"Wilcoxon p = {stats.wilcoxon(w2, w1).pvalue:.3g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Report per-paper "rather than" count statistics: the corpus-wide
maximum, how many papers clear one or more thresholds, and a top-N
table with each paper's raw count and rate per 1,000 words. Reproduces
the paper's "High-count papers" paragraph and Table 6 (checked
2026-09-14/15), and the "no ACL 2019 paper exceeds 12" comparison when
run against both corpora.

Usage:
    python3 high_count_report.py <text_dir> [--glob "*.txt"] \
        [--threshold 20 --threshold 35] [--top 7]
"""
import argparse
import re
from pathlib import Path

PHRASE_RE = re.compile(r"rather than", re.I)
WORD_RE = re.compile(r"\S+")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text_dir")
    p.add_argument("--glob", default="*.txt")
    p.add_argument("--threshold", type=int, action="append", default=[],
                    help="report how many papers reach >= this count; repeatable")
    p.add_argument("--top", type=int, default=7)
    args = p.parse_args()

    counts = {}
    words = {}
    for f in Path(args.text_dir).glob(args.glob):
        text = f.read_text(errors="replace")
        counts[f.stem] = len(PHRASE_RE.findall(text))
        words[f.stem] = len(WORD_RE.findall(text))

    n = len(counts)
    mx = max(counts.values())
    print(f"{n} papers; max count = {mx}")
    for t in args.threshold:
        q = [d for d, c in counts.items() if c >= t]
        print(f">= {t} occurrences: {len(q)} papers")

    print(f"\ntop {args.top} by raw count:")
    print(f"{'paper':<16}{'count':>6}{'/1k words':>12}")
    for doc in sorted(counts, key=lambda d: -counts[d])[:args.top]:
        rate = 1000 * counts[doc] / words[doc]
        print(f"{doc:<16}{counts[doc]:>6}{rate:>12.2f}")


if __name__ == "__main__":
    main()

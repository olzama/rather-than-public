#!/usr/bin/env python3
r"""
How much do papers whose back matter could not be cut affect the rates?
strip_backmatter.py keeps such papers whole. For each corpus, this splits
papers by whether the cut succeeded (the same test strip_backmatter.py
uses), and reports the mean "rather than" rate per 1,000 words (as in
count_antithesis_patterns.py) and mean body words, for all papers, papers
cut, and papers kept whole. Excluded documents are left out.

Usage (from counting/):
    python3 cut_sensitivity.py ../data --exclude ../data/excluded_documents.tsv
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import compile_patterns
from strip_backmatter import strip_acl2019, strip_arxiv

WORD_RE = re.compile(r"\w+")
CORPORA = {"acl2019": strip_acl2019, "arxiv2026": strip_arxiv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--exclude")
    args = ap.parse_args()
    rx = compile_patterns()["rather_than"]
    excluded = set()
    if args.exclude:
        excluded = {tuple(l.rstrip("\n").split("\t")[:2]) for l in list(open(args.exclude))[1:]}
    for corpus, strip in CORPORA.items():
        rows = []
        for f in sorted((Path(args.data_dir) / corpus / "text").glob("*.txt")):
            if (corpus, f.stem) in excluded:
                continue
            body, cut = strip(f.read_text(errors="replace"))
            n = len(WORD_RE.findall(body))
            if n:
                rows.append((cut, 1000 * len(rx.findall(body)) / n, n))
        for label, sel in (("all", rows), ("cut", [r for r in rows if r[0]]), ("kept whole", [r for r in rows if not r[0]])):
            print(f"{corpus:9s} {label:10s} papers {len(sel):5d}  rate {statistics.mean(r[1] for r in sel):.3f}"
                  f"  mean words {statistics.mean(r[2] for r in sel):.0f}")


if __name__ == "__main__":
    main()

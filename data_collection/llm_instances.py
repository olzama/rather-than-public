#!/usr/bin/env python3
r"""
Turn LLM-written papers (paper_pipeline output, Markdown) into plain text
laid out like the human corpora, so their "rather than" instances can be
extracted and shown on the annotation page without cues to their source.

Markdown formatting is removed (headings, emphasis, lists, tables, code,
links), and LaTeX math and author-year citations are replaced by
"[redacted]", the placeholder the arXiv corpora use for math and \cite.
One text file per paper and stage is written as
<out_dir>/<model>/<model>__<stage>__<paper_id>.txt; run
extract_antithesis_instances.py on each model folder with
--corpus-name llm_<model>.

Usage:
    python3 llm_instances.py ../data/generated/gpt_papers/batch2 ../data/generated/gpt_papers/batch3 --out-dir <out_dir> \
        [--models gpt-4o gpt-5.6-sol gpt-6-sol]
"""
import argparse
import re
from pathlib import Path

R = "[redacted]"
NAME = r"[A-Z][A-Za-z'\-]+(?: (?:et al\.|and [A-Z][A-Za-z'\-]+|& [A-Z][A-Za-z'\-]+))?"


def plain(t):
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\|.*\|\s*$", "", t, flags=re.M)
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", t)
    return re.sub(r"`([^`]*)`", r"\1", t)


def redact(t):
    t = re.sub(r"\$\$.*?\$\$", R, t, flags=re.S)
    t = re.sub(r"\\\[.*?\\\]", R, t, flags=re.S)
    t = re.sub(r"\\\(.*?\\\)", R, t, flags=re.S)
    t = re.sub(r"(?<!\\)\$[^$\n]{1,200}?\$", R, t)
    t = re.sub(r"\((?:[^()]*?\b\d{4}[a-z]?\b[^()]*?)\)",
               lambda m: R if re.search(NAME + r",? \d{4}", m.group(0)) else m.group(0), t)
    return re.sub(NAME + r" \(\d{4}[a-z]?\)", R, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generation_dirs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-5.6-sol", "gpt-6-sol"])
    args = ap.parse_args()
    for m in args.models:
        out = Path(args.out_dir) / m
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for stage in ("generated", "revised"):
            for f in sorted(f for gd in args.generation_dirs for f in (Path(gd) / m / f"{stage}_papers").glob("*.md")):
                (out / f"{m}__{stage}__{f.stem}.txt").write_text(redact(plain(f.read_text(errors="ignore"))))
                n += 1
        print(f"{m}: {n} files -> {out}")


if __name__ == "__main__":
    main()

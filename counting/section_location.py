#!/usr/bin/env python3
r"""
Where in a paper does "rather than" occur? Rate per 1,000 words by section
group, for original ACL 2019 papers (sections.jsonl.gz) and for LLM-written
versions of the same papers (Markdown, split at headings), generated and
revised. Tests whether revision adds the construction mainly in scoping
sections (abstract, limitations, conclusion), as a disavowal account
predicts.

Section groups: abstract; introduction; related work; body (method,
experiments, results, analysis, and sections with their own titles);
conclusion; limitations (limitations, ethics). In the Markdown papers a
subsection belongs to its parent section unless its own heading names a
group; the title line, references, acknowledgments and appendices are left
out, as are acknowledgments and appendices of the originals.

Usage:
    python3 section_location.py ../data/generated/gpt_papers/batch2 ../data/generated/gpt_papers/batch3 --acl-sections ../data/acl2019/sections.jsonl.gz \
        [--excluded ../data/excluded_documents.tsv] [--models gpt-5.6-sol gpt-6-sol]
"""
import argparse
import collections
import gzip
import json
import re
from pathlib import Path

RT = re.compile(r"\brather than\b", re.I)
WORD = re.compile(r"\w+")
GROUPS = ["abstract", "introduction", "related", "body", "conclusion", "limitations"]
SCOPING = ("abstract", "conclusion", "limitations")
RULES = [("abstract", r"abstract"), ("limitations", r"limitation|ethic|broader impact"),
         ("related", r"related|background|prior work|previous work"), ("introduction", r"introduction"),
         ("conclusion", r"conclusion|future work|summary"),
         ("drop", r"^references?$|bibliography|acknowledg|appendix")]
CAT_MAP = {"abstract": "abstract", "introduction": "introduction", "related_work": "related", "method": "body",
           "experiments": "body", "preliminaries": "body", "results": "body", "analysis": "body", "other": "body",
           "discussion": "body", "conclusion": "conclusion", "limitations": "limitations", "ethics": "limitations"}


def group(heading):
    h = re.sub(r"^[\d.\s]+", "", heading.lower()).strip()
    for g, pat in RULES:
        if re.search(pat, h):
            return g
    return None


def md_sections(text):
    """(group, text) pairs from Markdown headings (level 1 is the title)."""
    out, top, current, buf = [], "drop", "drop", []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            out.append((current, "\n".join(buf)))
            level, g = len(m.group(1)), group(m.group(2))
            if level == 1:
                top = current = g or "drop"
            elif level == 2:
                top = current = g or "body"
            else:
                current = g or top
            buf = []
        else:
            buf.append(line)
    out.append((current, "\n".join(buf)))
    return [(g, t) for g, t in out if g != "drop"]


def tally(pairs, counts):
    for g, t in pairs:
        if g not in GROUPS:
            continue
        counts[g][0] += len(RT.findall(t))
        counts[g][1] += len(WORD.findall(t))


def report(label, counts):
    tot_rt = sum(c[0] for c in counts.values())
    tot_w = sum(c[1] for c in counts.values())
    cells = []
    for g in GROUPS:
        rt, w = counts[g]
        cells.append(f"{1000 * rt / w:5.2f} ({100 * rt / max(tot_rt, 1):3.0f}%)" if w else "   -")
    sc_rt = sum(counts[g][0] for g in SCOPING)
    sc_w = sum(counts[g][1] for g in SCOPING)
    print(f"{label:24s} {1000 * tot_rt / tot_w:5.2f} " + " ".join(f"{c:>12s}" for c in cells)
          + f"   scoping: {100 * sc_rt / max(tot_rt, 1):3.0f}% of uses, {100 * sc_w / tot_w:3.0f}% of words")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generation_dirs", nargs="+", help="one or more generation runs (data/generated/gpt_papers/batch*)")
    ap.add_argument("--acl-sections", required=True, help="ACL 2019 sections.jsonl.gz")
    ap.add_argument("--excluded")
    ap.add_argument("--models", nargs="+", default=["gpt-5.6-sol", "gpt-6-sol"])
    args = ap.parse_args()
    from generation_view import combined
    g = combined(args.generation_dirs, args.models)
    skip = set()
    if args.excluded:
        skip = {l.split("\t")[1] for l in list(open(args.excluded))[1:] if l.split("\t")[0] == "acl2019"}
    ids = set.intersection(*({p.stem for p in (g / m / s).glob("*.md")} for m in args.models
                             for s in ("generated_papers", "revised_papers"))) - skip
    orig = collections.defaultdict(lambda: [0, 0])
    for line in gzip.open(args.acl_sections, "rt"):
        r = json.loads(line)
        if r["doc_id"] in ids:
            tally([(CAT_MAP.get(s["category"], "drop"), s["text"]) for s in r["sections"]], orig)
    print(f"{len(ids)} papers; rate per 1,000 words (share of the source's uses)")
    print(f"{'source':24s} {'all':>5s} " + " ".join(f"{x:>12s}" for x in GROUPS))
    report("ACL 2019 original", orig)
    for m in args.models:
        got = {}
        for s in ("generated_papers", "revised_papers"):
            c = collections.defaultdict(lambda: [0, 0])
            for i in ids:
                tally(md_sections((g / m / s / f"{i}.md").read_text(errors="replace")), c)
            got[s] = report(f"{m} {s.split('_')[0]}", c)
        added = {x: got["revised_papers"][x][0] - got["generated_papers"][x][0] for x in GROUPS}
        print(f"{'  added by revision':24s}       " + " ".join(f"{added[x]:>12d}" for x in GROUPS))


if __name__ == "__main__":
    main()

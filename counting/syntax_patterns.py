#!/usr/bin/env python3
r"""
Which syntactic patterns of "rather than" sentences go with annoyance?
Every pattern found in the parsed sentences (spaCy) is collected, without a
preselected list, and each pattern that at least --min items have is tested
for association with the label (annoying vs legitimate).

Patterns per sentence (X and Y as in syntax_features.tree_xy):
  frame:   attachment of the construction: dependency label of "than", the
           tag of the word it attaches to, and of "rather"
  pos:     "rather than" opens the sentence
  X:, Y:   tag and dependency label of the X and Y heads; XY: their tag pair
  root:    tag of the main-clause root; subj: its subject (syntax_features)
  tag:     every part-of-speech tag in the sentence
  dep:     every dependency label in the sentence
  arc:     every head-tag --label--> dependent-tag arc in the sentence
A pattern counts once per sentence. Binary association is tested with
Fisher's exact test (Cochran-Mantel-Haenszel over corpora with --stratify);
Benjamini-Hochberg q-values are computed across all tested patterns.

Usage:
    python3 syntax_patterns.py ../data/annotation [--annotator A1] [--corpora ...] \
        [--min 5] [--stratify] [--batch 1|2] [--model en_core_web_sm] [--top 30] [--out tested.tsv]
"""
import argparse
import collections
import json
import re
from pathlib import Path

import spacy
from scipy import stats

from syntax_features import ARXIV2026, bh, cmh_p, main_clause, subject_type, tree_xy, head_of


def patterns(nlp, it):
    s = it["sentence"]
    doc = nlp(s)
    P = set()
    if not re.search(r"[A-Za-z0-9]", s[:it["local_start"]]):
        P.add("pos:initial")
    than = next((t for t in doc if t.lower_ == "than" and t.idx >= it["local_start"]), None)
    rather = next((t for t in doc if t.lower_ == "rather" and t.idx >= it["local_start"] - 1), None)
    if than is not None:
        P.add(f"frame:than={than.dep_}<-{than.head.tag_}")
    if rather is not None:
        P.add(f"frame:rather={rather.dep_}<-{rather.head.tag_}")
    x, y = tree_xy(doc, it["local_start"])
    hx, hy = (head_of(x) if x else None), (head_of(y) if y else None)
    for name, h in (("X", hx), ("Y", hy)):
        if h is not None:
            P.add(f"{name}:tag={h.tag_}")
            P.add(f"{name}:dep={h.dep_}")
    if hx is not None and hy is not None:
        P.add(f"XY:{hx.tag_}/{hy.tag_}")
    root = main_clause(doc, it["local_start"])
    P.add(f"root:{root.tag_}")
    P.add(f"subj:{subject_type(root)}")
    for t in doc:
        if t.is_space:
            continue
        P.add(f"tag:{t.tag_}")
        if t.head.i != t.i:
            P.add(f"dep:{t.dep_}")
            P.add(f"arc:{t.head.tag_}-{t.dep_}->{t.tag_}")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--corpora", nargs="+", default=sorted(ARXIV2026))
    ap.add_argument("--annotator", default="A1")
    ap.add_argument("--min", type=int, default=5, help="test patterns that at least this many items have")
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--model", default="en_core_web_sm")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", help="write every tested pattern as TSV")
    ap.add_argument("--batch", type=int, help="only items of this annotation batch (items_public.jsonl)")
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    nlp = spacy.load(args.model)
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl")) if "excluded" not in r}
    pub = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl")) if r["item_id"] in prov}
    labels = {r["item_id"]: r["label"] for r in map(json.loads, open(d / "human_labels.jsonl")) if r["annotator"] == args.annotator}
    items = sorted(i for i in pub if prov[i]["corpus"] in args.corpora and "repeat_of" not in prov[i]
                   and labels.get(i) in ("annoying", "legitimate")
                   and (args.batch is None or pub[i].get("batch", 1) == args.batch))
    rows = [(prov[i]["corpus"], labels[i] == "annoying", patterns(nlp, pub[i])) for i in items]
    n_ann = sum(a for _, a, _ in rows); n_leg = len(rows) - n_ann
    count = collections.Counter(p for *_, P in rows for p in P)
    tests = []
    for p, n in count.items():
        if n < args.min or n > len(rows) - args.min:
            continue
        a = sum(ann and p in P for _, ann, P in rows); b = n - a
        pv = (cmh_p([(c, ann, p in P) for c, ann, P in rows], None) if args.stratify
              else stats.fisher_exact([[a, n_ann - a], [b, n_leg - b]])[1])
        tests.append((p, a, b, pv))
    qs = bh([t[3] for t in tests])
    res = sorted(zip(tests, qs), key=lambda r: r[0][3])
    fam = collections.Counter(t[0].split(":")[0] for t in tests)
    print(f"{args.annotator}, {', '.join(args.corpora)}: {n_ann} annoying, {n_leg} legitimate; "
          f"{len(count)} patterns found, {len(tests)} tested (>= {args.min} items): "
          + ", ".join(f"{k} {v}" for k, v in sorted(fam.items())))
    print(f"q < .05: {sum(q < .05 for q in qs)}; q < .10: {sum(q < .10 for q in qs)}\n")
    print(f"{'pattern':40s} {'annoying':>12s} {'legitimate':>12s} {'p':>8s} {'q':>6s}")
    for (p, a, b, pv), q in res[:args.top]:
        print(f"{p:40s} {a:4d} ({a / n_ann:4.0%}) {b:4d} ({b / n_leg:4.0%}) {pv:8.2g} {q:6.2f}")
    if args.out:
        with open(args.out, "w") as f:
            f.write("pattern\tannoying\tlegitimate\tn_annoying\tn_legitimate\tp\tq\n")
            for (p, a, b, pv), q in res:
                f.write(f"{p}\t{a}\t{b}\t{n_ann}\t{n_leg}\t{pv:.4g}\t{q:.4g}\n")


if __name__ == "__main__":
    main()

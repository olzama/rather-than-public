#!/usr/bin/env python3
r"""
Find "X-ed rather than Y-ed": "rather than" with a past participle
immediately before "rather" and immediately after "than" (spaCy tag VBN),
e.g. "counted rather than assumed", "disclosed rather than defended". This
is the surface form of the clearest straw men both coders agreed on.

For each instance file (per-instance output of
extract_antithesis_instances.py), prints the number of "rather than"
instances, of matches, the share, and papers with at least one match, and
writes every match (corpus, doc_id, X, Y, sentence) to --out. Documents in
--exclude (language_filter.py output) are skipped.

Usage:
    python3 participle_contrast.py --instances acl2019=../data/antithesis_instances/acl2019.jsonl \
        --instances arxiv2026=../data/antithesis_instances/arxiv2026.jsonl \
        [--instances gpt-6-sol/generated=.../antithesis_instances.jsonl ...] \
        [--exclude ../data/excluded_documents.tsv] [--out matches.jsonl]
"""
import argparse
import collections
import json
import re

import spacy

# exclusion-list corpus for each label; generated papers carry their source paper's id
LIST_CORPUS = {"acl2019": "acl2019", "arxiv2026": "arxiv2026", "arxiv_v1": "arxiv_versions",
               "arxiv_latest": "arxiv_versions"}


def list_corpus(label, doc_id):
    if label in LIST_CORPUS:
        return LIST_CORPUS[label]
    return "arxiv2026" if re.match(r"^\d{4}\.\d{4,5}$", doc_id) else "acl2019"


def match(nlp_doc, local_start, local_end):
    toks = list(nlp_doc)
    rather = next((k for k, t in enumerate(toks) if t.idx >= local_start and t.lower_ == "rather"), None)
    if rather is None or rather + 2 >= len(toks) or toks[rather + 1].lower_ != "than" or rather == 0:
        return None
    x, y = toks[rather - 1], toks[rather + 2]
    if x.tag_ == "VBN" and y.tag_ == "VBN":
        return x.text, y.text
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", action="append", required=True, help="LABEL=PATH")
    ap.add_argument("--exclude")
    ap.add_argument("--out")
    ap.add_argument("--model", default="en_core_web_sm")
    args = ap.parse_args()
    nlp = spacy.load(args.model, disable=["ner", "lemmatizer"])
    skip = set()
    if args.exclude:
        for line in list(open(args.exclude))[1:]:
            c, doc = line.split("\t")[:2]
            skip.add((c, doc))
    out = open(args.out, "w") if args.out else None
    print(f"{'corpus':28s} {'RT':>7s} {'X-ed/Y-ed':>9s} {'share':>7s} {'papers':>7s} {'with':>6s}")
    for spec in args.instances:
        label, path = spec.split("=", 1)
        recs = [r for r in map(json.loads, open(path)) if r["pattern_id"] == "rather_than"
                and (list_corpus(label, r["doc_id"]), r["doc_id"]) not in skip]
        docs = nlp.pipe((r["sentence"] for r in recs), batch_size=256)
        n, hits, papers, with_hit = 0, 0, set(), set()
        for r, doc in zip(recs, docs):
            n += 1
            papers.add(r["doc_id"])
            m = match(doc, r["local_start"], r["local_end"])
            if m:
                hits += 1
                with_hit.add(r["doc_id"])
                if out:
                    out.write(json.dumps({"corpus": label, "doc_id": r["doc_id"], "x": m[0], "y": m[1],
                                          "sentence": r["sentence"]}) + "\n")
        print(f"{label:28s} {n:7d} {hits:9d} {hits / max(n, 1):7.2%} {len(papers):7d} {len(with_hit):6d}")
    if out:
        out.close()


if __name__ == "__main__":
    main()

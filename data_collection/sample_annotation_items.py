#!/usr/bin/env python3
r"""
Draw a stratified, blinded sample of "rather than" instances for the
legitimate-vs-annoying annotation task (agreed with the user
2026-09-14): a few hundred instances, drawn evenly across corpora, mixed
together so no annotator (human or LLM) can tell which corpus an item
came from, with provenance kept separately so it can be restored later.

Reads the four antithesis_instances/*.jsonl files already produced by
extract_antithesis_instances.py, filters to pattern_id == "rather_than",
and draws --per-corpus items from each at random (seeded, reproducible).

Usage:
    python3 sample_annotation_items.py <instances_dir> <out_dir> \
        [--per-corpus 75] [--seed 20260914]

<instances_dir> holds acl2019.jsonl, arxiv2026.jsonl, arxiv_v1.jsonl,
arxiv_latest.jsonl (the "corpus" field inside each record is what's
stratified on, not the filename, so this also works if corpora are
combined into fewer files later). Writes two files to <out_dir>:

    items_public.jsonl     {item_id, sentence}            -- show to raters
    items_provenance.jsonl {item_id, corpus, doc_id,       -- keep private
                             pattern_id, char_start, char_end, match_text}

item_id is assigned AFTER shuffling the pooled sample, so sequential ids
carry no information about source or draw order.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from excluded_docs import drop_excluded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("instances_dir")
    p.add_argument("out_dir")
    p.add_argument("--per-corpus", type=int, default=75)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--pattern", default="rather_than")
    args = p.parse_args()

    instances_dir = Path(args.instances_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for f in sorted(instances_dir.glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["pattern_id"] == args.pattern:
                records.append(r)
    by_corpus = defaultdict(list)
    for r in drop_excluded(records):
        by_corpus[r["corpus"]].append(r)

    if not by_corpus:
        sys.exit(f"no instances with pattern_id={args.pattern} found under {instances_dir}")

    rng = random.Random(args.seed)
    pooled = []
    for corpus, records in sorted(by_corpus.items()):
        k = min(args.per_corpus, len(records))
        if k < args.per_corpus:
            print(f"WARNING: {corpus} only has {len(records)} instances, taking all of them", file=sys.stderr)
        pooled.extend(rng.sample(records, k))
        print(f"{corpus}: {len(records)} available, sampled {k}", file=sys.stderr)

    rng.shuffle(pooled)

    n = len(pooled)
    width = len(str(n))
    public_path = out_dir / "items_public.jsonl"
    provenance_path = out_dir / "items_provenance.jsonl"
    with open(public_path, "w") as pub, open(provenance_path, "w") as prov:
        for i, r in enumerate(pooled, 1):
            item_id = f"item_{i:0{width}d}"
            pub.write(json.dumps({"item_id": item_id, "sentence": r["sentence"]}) + "\n")
            prov.write(json.dumps({
                "item_id": item_id, "corpus": r["corpus"], "doc_id": r["doc_id"],
                "pattern_id": r["pattern_id"], "char_start": r["char_start"],
                "char_end": r["char_end"], "match_text": r["match_text"],
            }) + "\n")

    print(f"DONE {n} items -> {public_path} (+ {provenance_path}, keep private)", file=sys.stderr)


if __name__ == "__main__":
    main()

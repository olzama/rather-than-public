#!/usr/bin/env python3
r"""
Append a new batch of "rather than" instances to an existing blinded
annotation sample (items_public.jsonl / items_provenance.jsonl),
excluding every instance already in it -- including the same-instance-
under-a-different-corpus-label case (see resize_annotation_sample.py's
docstring for why that matters: dataset (2) reuses the "latest" version
text for papers also in the version-diff subset, so a candidate drawn
from arxiv2026 can be the exact same sentence as one already used from
arxiv_v1/arxiv_latest in an earlier round).

Usage:
    python3 add_annotation_batch.py <instances_dir> <out_dir> \
        --corpus acl2019:50 --corpus arxiv2026:50 [--seed 20260914] [--batch 2]

Appends to <out_dir>/items_public.jsonl and items_provenance.jsonl.
New item_ids start one past the highest numeric id in the pool, the
provenance file, repeats.jsonl and human_labels.jsonl, so they cannot
collide with any id that already has labels. With --batch N, new public
items carry "batch": N; the annotation page orders and repeats each batch
separately, so adding a batch leaves earlier batches' order unchanged.
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from excluded_docs import drop_excluded

VERSION_SUFFIX_RE = re.compile(r"_v\d+$")
ITEM_NUM_RE = re.compile(r"item_(\d+)$")


def canonical_doc(doc_id):
    return VERSION_SUFFIX_RE.sub("", doc_id)


def instance_key(doc_id, sentence, local_start):
    return (canonical_doc(doc_id), sentence, local_start)


def load_instances(instances_dir, pattern="rather_than"):
    records = []
    for f in sorted(Path(instances_dir).glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["pattern_id"] == pattern:
                records.append(r)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("instances_dir")
    p.add_argument("out_dir")
    p.add_argument("--corpus", action="append", required=True, help="CORPUS:COUNT, repeatable")
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--batch", type=int, help="batch number stored on the new public items")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    pub_path = out_dir / "items_public.jsonl"
    prov_path = out_dir / "items_provenance.jsonl"

    existing_pub = [json.loads(l) for l in open(pub_path)]
    existing_prov = [json.loads(l) for l in open(prov_path)]
    prov_by_id = {r["item_id"]: r for r in existing_prov}

    used_keys = set()
    max_num = 0
    for pub in existing_pub:
        prov = prov_by_id[pub["item_id"]]
        used_keys.add(instance_key(prov["doc_id"], pub["sentence"], pub["local_start"]))
        m = ITEM_NUM_RE.match(pub["item_id"])
        if m:
            max_num = max(max_num, int(m.group(1)))

    # ids used in sibling pool folders (e.g. ../annotation_b3) must not be reused either
    id_files = [out_dir / f for f in ("items_provenance.jsonl", "repeats.jsonl", "human_labels.jsonl")]
    id_files += sorted(out_dir.parent.glob("annotation*/items_public.jsonl"))
    for path in id_files:
        extra = path.name
        if path.exists():
            for line in open(path):
                m = ITEM_NUM_RE.search(json.loads(line)["item_id"])
                if m:
                    max_num = max(max_num, int(m.group(1)))

    targets = []
    for spec in args.corpus:
        corpus, count = spec.rsplit(":", 1)
        targets.append((corpus, int(count)))

    all_instances = drop_excluded(load_instances(args.instances_dir))
    by_corpus = defaultdict(list)
    for r in all_instances:
        by_corpus[r["corpus"]].append(r)

    rng = random.Random(args.seed)
    new_records = []
    for corpus, count in targets:
        candidates = by_corpus.get(corpus, [])
        rng.shuffle(candidates)
        picked = []
        for r in candidates:
            if len(picked) >= count:
                break
            key = instance_key(r["doc_id"], r["sentence"], r["local_start"])
            if key in used_keys:
                continue
            used_keys.add(key)
            picked.append(r)
        if len(picked) < count:
            print(f"WARNING: {corpus} only yielded {len(picked)}/{count} new candidates", file=sys.stderr)
        new_records.extend(picked)
        print(f"{corpus}: picked {len(picked)} new", file=sys.stderr)

    next_num = max_num + 1
    new_items = []
    for i, rec in enumerate(new_records):
        new_items.append((f"item_{next_num + i}", rec))

    rng.shuffle(new_items)

    with open(pub_path, "a") as pub, open(prov_path, "a") as prov_out:
        for item_id, rec in new_items:
            item = {
                "item_id": item_id, "sentence": rec["sentence"],
                "local_start": rec["local_start"], "local_end": rec["local_end"],
                "context_before": rec.get("context_before", ""),
                "context_after": rec.get("context_after", ""),
            }
            if args.batch is not None:
                item["batch"] = args.batch
            pub.write(json.dumps(item) + "\n")
            prov_out.write(json.dumps({
                "item_id": item_id, "corpus": rec["corpus"], "doc_id": rec["doc_id"],
                "pattern_id": rec["pattern_id"], "char_start": rec["char_start"],
                "char_end": rec["char_end"], "match_text": rec["match_text"],
            }) + "\n")

    print(f"DONE appended {len(new_items)} items -> {pub_path} "
          f"(total now {len(existing_pub) + len(new_items)})", file=sys.stderr)


if __name__ == "__main__":
    main()

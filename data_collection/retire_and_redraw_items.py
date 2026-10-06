#!/usr/bin/env python3
r"""
Retire specific annotation-pool items and replace them with fresh draws
from a (corrected) instances file. Generalizes the original
redo_garbled_items.py: that script could only select items by a DB
label (e.g. "garbled"); this one also accepts an explicit item_id list,
for cases like "these items were drawn before an extraction fix and
are stale regardless of what they were labeled" -- e.g. the 43
ACL 2019 pool items drawn before the pdftotext -layout fix that happen
not to have been marked garbled, or arXiv items affected by detex
silently dropping math/citations.

Removes the given item_ids from items_public.jsonl / items_provenance.jsonl
(their DB answer rows are untouched -- they just no longer point at a
live pool item) and appends an equal number of new instances of
--pattern from --corpus, respecting the same canonical dedup key used
throughout this project's sampling scripts (doc_id with any _vN suffix
stripped, sentence text, local_start).

Usage:
    # by explicit item_id list (one per line)
    python3 retire_and_redraw_items.py <out_dir> --instances-dir <dir> --corpus acl2019 \
        --item-ids-file ids.txt

    # by DB label (original use case)
    python3 retire_and_redraw_items.py <out_dir> --instances-dir <dir> --corpus acl2019 \
        --labels-path human_labels.jsonl --label garbled
"""
import argparse
import glob as globmod
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from excluded_docs import drop_excluded

VERSION_SUFFIX_RE = re.compile(r"_v\d+$")
ITEM_NUM_RE = re.compile(r"item_(\d+)$")
PHRASE_RE = re.compile(r"rather than", re.I)


def canonical_doc(doc_id):
    return VERSION_SUFFIX_RE.sub("", doc_id)


def instance_key(doc_id, sentence, local_start):
    return (canonical_doc(doc_id), sentence, local_start)


def load_instances(instances_dir, corpus, pattern="rather_than"):
    records = []
    path = Path(instances_dir) / f"{corpus}.jsonl"
    for line in open(path):
        r = json.loads(line)
        if r["pattern_id"] == pattern:
            records.append(r)
    return records


def paper_counts(text_dir):
    counts = {}
    for f in globmod.glob(f"{text_dir}/*.txt"):
        text = open(f, errors="replace").read()
        counts[Path(f).stem] = len(PHRASE_RE.findall(text))
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--instances-dir", required=True)
    p.add_argument("--corpus", required=True, help="which corpus's instances/*.jsonl to draw replacements from")
    p.add_argument("--pattern", default="rather_than")
    p.add_argument("--item-ids-file", help="text file, one item_id per line, to retire")
    p.add_argument("--labels-path", help="human_labels.jsonl, used with --label")
    p.add_argument("--label", help="retire every live item with this label in --labels-path")
    p.add_argument("--corpus-label", default=None,
                    help="corpus value to write into provenance for new items, if different from --corpus "
                         "(e.g. --corpus arxiv2026 --corpus-label high_count_arxiv2026)")
    p.add_argument("--min-paper-count", type=int, default=None,
                    help="only draw replacements from papers whose --text-dir occurrence count is >= this "
                         "(for redrawing a high-count stratum)")
    p.add_argument("--text-dir", default=None, help="text/ dir to compute per-paper counts from, used with --min-paper-count")
    p.add_argument("--max-per-paper", type=int, default=None,
                    help="cap instances per paper across the WHOLE resulting --corpus-label pool, used with --min-paper-count")
    p.add_argument("--seed", type=int, default=20260914)
    args = p.parse_args()

    if not args.item_ids_file and not (args.labels_path and args.label):
        sys.exit("need either --item-ids-file or (--labels-path and --label)")

    out_dir = Path(args.out_dir)
    pub_path = out_dir / "items_public.jsonl"
    prov_path = out_dir / "items_provenance.jsonl"

    existing_pub = [json.loads(l) for l in open(pub_path)]
    existing_prov = [json.loads(l) for l in open(prov_path)]
    prov_by_id = {r["item_id"]: r for r in existing_prov}

    to_remove = set()
    if args.item_ids_file:
        to_remove |= {l.strip() for l in open(args.item_ids_file) if l.strip()}
    if args.labels_path and args.label:
        to_remove |= {
            json.loads(l)["item_id"] for l in open(args.labels_path)
            if json.loads(l)["label"] == args.label
        }

    # only remove ones actually still live in the pool
    live_ids = {r["item_id"] for r in existing_pub}
    unknown = to_remove - live_ids
    if unknown:
        print(f"NOTE: {len(unknown)} requested item_ids are not live pool items, skipping: {sorted(unknown)}", file=sys.stderr)
    to_remove &= live_ids
    if not to_remove:
        sys.exit("no live pool items to retire -- nothing to do")
    print(f"retiring {len(to_remove)} items: {sorted(to_remove)}", file=sys.stderr)

    kept_pub = [r for r in existing_pub if r["item_id"] not in to_remove]
    kept_prov = [r for r in existing_prov if r["item_id"] not in to_remove]

    used_keys = set()
    max_num = 0
    for pub in existing_pub:  # dedupe against the FULL pre-removal pool
        prov = prov_by_id[pub["item_id"]]
        used_keys.add(instance_key(prov["doc_id"], pub["sentence"], pub["local_start"]))
        m = ITEM_NUM_RE.match(pub["item_id"])
        if m:
            max_num = max(max_num, int(m.group(1)))

    count = len(to_remove)
    corpus_label = args.corpus_label or args.corpus
    candidates = drop_excluded(load_instances(args.instances_dir, args.corpus, args.pattern))

    per_paper_taken = defaultdict(int)
    counts_by_paper = None
    if args.min_paper_count is not None:
        if not args.text_dir:
            sys.exit("--min-paper-count requires --text-dir")
        counts_by_paper = paper_counts(args.text_dir)
        qualifying = {d for d, c in counts_by_paper.items() if c >= args.min_paper_count}
        candidates = [r for r in candidates if r["doc_id"] in qualifying]
        # per-paper usage already committed to this corpus_label, across the KEPT pool
        # (retired items free up their slot; other still-live items still count)
        for pub in kept_pub:
            prov = prov_by_id.get(pub["item_id"])
            if prov and prov["corpus"] == corpus_label:
                per_paper_taken[prov["doc_id"]] += 1

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    picked = []
    for r in candidates:
        if len(picked) >= count:
            break
        key = instance_key(r["doc_id"], r["sentence"], r["local_start"])
        if key in used_keys:
            continue
        if args.max_per_paper is not None and per_paper_taken[r["doc_id"]] >= args.max_per_paper:
            continue
        used_keys.add(key)
        picked.append(r)
        per_paper_taken[r["doc_id"]] += 1

    if len(picked) < count:
        print(f"WARNING: only found {len(picked)}/{count} eligible replacement instances "
              f"(qualifying papers / per-paper cap may be exhausted)", file=sys.stderr)

    next_num = max_num + 1
    new_items = []
    for i, rec in enumerate(picked):
        new_items.append((f"item_{next_num + i}", rec))

    for item_id, rec in new_items:
        kept_pub.append({
            "item_id": item_id, "sentence": rec["sentence"],
            "local_start": rec["local_start"], "local_end": rec["local_end"],
            "context_before": rec.get("context_before", ""),
            "context_after": rec.get("context_after", ""),
        })
        prov_rec = {
            "item_id": item_id, "corpus": corpus_label, "doc_id": rec["doc_id"],
            "pattern_id": rec["pattern_id"], "char_start": rec["char_start"],
            "char_end": rec["char_end"], "match_text": rec["match_text"],
        }
        if counts_by_paper is not None:
            prov_rec["paper_total_count"] = counts_by_paper[rec["doc_id"]]
        kept_prov.append(prov_rec)

    with open(pub_path, "w") as pub:
        for r in kept_pub:
            pub.write(json.dumps(r) + "\n")
    with open(prov_path, "w") as prov_out:
        for r in kept_prov:
            prov_out.write(json.dumps(r) + "\n")

    print(f"DONE retired {len(to_remove)}, added {len(new_items)} -> pool size now {len(kept_pub)} "
          f"(was {len(existing_pub)})", file=sys.stderr)
    print(f"retired item_ids: {sorted(to_remove)}", file=sys.stderr)
    print(f"new item_ids: {[iid for iid, _ in new_items]}", file=sys.stderr)


if __name__ == "__main__":
    main()

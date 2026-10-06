#!/usr/bin/env python3
r"""
Rebuild the blinded annotation sample at a new target size, preserving
already-collected answers and fixing two bugs found during real use
(2026-09-14):

1. A sentence can contain more than one "rather than" -- each needs a
   separate judgment, but nothing told the UI which occurrence a given
   item was about, so it highlighted every occurrence indiscriminately.
   Fixed upstream in extract_antithesis_instances.py (local_start/
   local_end per match); this script's job is to make sure the
   *sample* still keeps distinct occurrences as distinct items.

2. dataset (2)'s corpus reuses the "latest" version source for the
   ~807 papers also in the version-diff subset (see
   build_arxiv2026_corpus.py --reuse-dir), so the exact same sentence
   can get drawn once as "arxiv2026" and again as "arxiv_latest" (or
   "arxiv_v1", if that version happens to share the sentence). These
   are the same real-world instance, not two independent data points,
   and must be deduplicated -- unlike case 1, where two matches in one
   sentence(same corpus) genuinely are two different data points.

The distinguishing signal: same canonical paper (doc_id with any
"_vN" suffix stripped) + identical sentence text + identical
local_start is the *same* match surfacing under two corpus labels
(dedupe, keep one); identical sentence text with a *different*
local_start is two real occurrences in one sentence (keep both).

Usage:
    python3 resize_annotation_sample.py <instances_dir> <old_out_dir> <answers.json> <new_out_dir> \
        [--target 100] [--seed 20260914]

<answers.json>: {item_id: label, ...} pulled from the live annotation
db -- these are preserved (same item_id, so existing saved rows in the
Artifact's database stay valid) unless dropped as an exact duplicate of
another preserved item (reported on stderr). New item_ids for freshly
drawn items start at "item_1001" so they can never collide with the
old 1-300 numbering.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from excluded_docs import drop_excluded

VERSION_SUFFIX_RE = re.compile(r"_v\d+$")


def canonical_doc(doc_id):
    return VERSION_SUFFIX_RE.sub("", doc_id)


def load_instances(instances_dir, pattern="rather_than"):
    """Return list of all rather_than instance records across corpora."""
    records = []
    for f in sorted(Path(instances_dir).glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["pattern_id"] == pattern:
                records.append(r)
    return records


def instance_key(r):
    """Dedup key: same underlying sentence occurrence, regardless of which
    corpus label it surfaced under."""
    return (canonical_doc(r["doc_id"]), r["sentence"], r["local_start"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("instances_dir")
    p.add_argument("old_out_dir")
    p.add_argument("answers_path")
    p.add_argument("new_out_dir")
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260914)
    args = p.parse_args()

    old_out_dir = Path(args.old_out_dir)
    new_out_dir = Path(args.new_out_dir)
    new_out_dir.mkdir(parents=True, exist_ok=True)

    answers = json.load(open(args.answers_path))
    old_provenance = {json.loads(l)["item_id"]: json.loads(l) for l in open(old_out_dir / "items_provenance.jsonl")}

    all_instances = load_instances(args.instances_dir)
    # index by (corpus, doc_id, char_start, char_end) to resolve old provenance -> regenerated record
    by_old_key = {(r["corpus"], r["doc_id"], r["char_start"], r["char_end"]): r for r in all_instances}

    used_keys = set()
    preserved = []  # (item_id, label, record)
    dropped = []
    for item_id, label in answers.items():
        prov = old_provenance.get(item_id)
        if prov is None:
            print(f"WARNING: answered {item_id} has no old provenance entry, skipping", file=sys.stderr)
            continue
        rec = by_old_key.get((prov["corpus"], prov["doc_id"], prov["char_start"], prov["char_end"]))
        if rec is None:
            print(f"WARNING: {item_id} not found in regenerated instances, skipping", file=sys.stderr)
            continue
        key = instance_key(rec)
        if key in used_keys:
            dropped.append((item_id, label, rec))
            continue
        used_keys.add(key)
        preserved.append((item_id, label, rec))

    print(f"{len(answers)} answered; {len(preserved)} preserved as distinct instances; "
          f"{len(dropped)} dropped as exact duplicates of another preserved item:", file=sys.stderr)
    for item_id, label, rec in dropped:
        print(f"  {item_id} ({label}) duplicates an already-preserved instance -- {rec['corpus']}/{rec['doc_id']}", file=sys.stderr)

    remaining_needed = max(0, args.target - len(preserved))
    print(f"target={args.target}, need {remaining_needed} more", file=sys.stderr)

    # pool candidates for the remaining slots: everything not already used
    by_corpus = defaultdict(list)
    for r in drop_excluded(all_instances):
        if instance_key(r) in used_keys:
            continue
        by_corpus[r["corpus"]].append(r)

    import random
    rng = random.Random(args.seed)
    corpora = sorted(by_corpus)
    per_corpus_target = remaining_needed // len(corpora)
    extra = remaining_needed - per_corpus_target * len(corpora)

    new_picks = []
    for i, corpus in enumerate(corpora):
        k = per_corpus_target + (1 if i < extra else 0)
        candidates = by_corpus[corpus]
        rng.shuffle(candidates)
        picked_this_corpus = []
        for r in candidates:
            if len(picked_this_corpus) >= k:
                break
            key = instance_key(r)
            if key in used_keys:
                continue
            used_keys.add(key)
            picked_this_corpus.append(r)
        if len(picked_this_corpus) < k:
            print(f"WARNING: {corpus} only yielded {len(picked_this_corpus)}/{k} new candidates", file=sys.stderr)
        new_picks.extend(picked_this_corpus)
        print(f"{corpus}: picked {len(picked_this_corpus)} new", file=sys.stderr)

    # assign fresh ids to new picks, starting at item_1001 (well clear of the old 1-300 range)
    new_items = []
    for i, rec in enumerate(new_picks, 1001):
        new_items.append((f"item_{i}", None, rec))

    final = preserved + new_items
    rng.shuffle(final)

    with open(new_out_dir / "items_public.jsonl", "w") as pub, \
         open(new_out_dir / "items_provenance.jsonl", "w") as prov_out:
        for item_id, label, rec in final:
            pub.write(json.dumps({
                "item_id": item_id, "sentence": rec["sentence"],
                "local_start": rec["local_start"], "local_end": rec["local_end"],
            }) + "\n")
            prov_out.write(json.dumps({
                "item_id": item_id, "corpus": rec["corpus"], "doc_id": rec["doc_id"],
                "pattern_id": rec["pattern_id"], "char_start": rec["char_start"],
                "char_end": rec["char_end"], "match_text": rec["match_text"],
            }) + "\n")

    preserved_ids = {item_id for item_id, _, _ in preserved}
    print(f"DONE {len(final)} items -> {new_out_dir} ({len(preserved)} preserved, {len(new_items)} new)", file=sys.stderr)
    print("PRESERVED_ITEM_IDS=" + ",".join(sorted(preserved_ids)), file=sys.stderr)


if __name__ == "__main__":
    main()

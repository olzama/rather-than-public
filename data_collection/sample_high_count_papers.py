#!/usr/bin/env python3
r"""
Draw a new annotation batch specifically from "high-count" papers: those
whose raw \emph{rather than} count is far above typical (ACL 2019's
maximum across all 4,863 papers is 12; the threshold here, 20, is
already above that, and arXiv 2026 has 124 papers clearing it, up to 69
in one paper). This is a third sampling stratum
alongside the existing per-corpus (2019 / 2026) draws -- not "more 2026
papers," but specifically papers that lean unusually hard on the
construction, to test whether that concentration predicts a higher
annoying-vs-legitimate rate.

Excludes every instance already in the pool (same canonical dedup key
as add_annotation_batch.py / resize_annotation_sample.py), and caps how
many instances come from any single paper (--max-per-paper) so the
batch isn't dominated by the single most extreme outlier.

Usage:
    python3 sample_high_count_papers.py <instances_dir> <text_dir> <out_dir> \
        --corpus-name arxiv2026 --threshold 20 --count 100 --max-per-paper 5 \
        [--seed 20260914]

Appends to <out_dir>/items_public.jsonl and items_provenance.jsonl, with
corpus label "high_count_<corpus-name>" in the provenance file (the
public file stays blind as always). New item_ids start one past the
current max.
"""
import argparse
import glob
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


def paper_counts(text_dir):
    counts = {}
    for f in glob.glob(f"{text_dir}/*.txt"):
        text = open(f, errors="replace").read()
        counts[Path(f).stem] = len(PHRASE_RE.findall(text))
    return counts


def load_rather_than_instances(instances_dir, corpus_name):
    records = []
    path = Path(instances_dir) / f"{corpus_name}.jsonl"
    for line in open(path):
        r = json.loads(line)
        if r["pattern_id"] == "rather_than":
            records.append(r)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("instances_dir")
    p.add_argument("text_dir")
    p.add_argument("out_dir")
    p.add_argument("--corpus-name", required=True, help="which corpus's instances/*.jsonl and text/ to draw from")
    p.add_argument("--threshold", type=int, default=20)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--max-per-paper", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260914)
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

    counts = paper_counts(args.text_dir)
    qualifying = {doc for doc, c in counts.items() if c >= args.threshold}
    print(f"{len(qualifying)} papers in {args.corpus_name} have >= {args.threshold} occurrences "
          f"(total {sum(counts[d] for d in qualifying)} instances)", file=sys.stderr)
    if not qualifying:
        sys.exit("no qualifying papers -- lower --threshold")

    instances = drop_excluded(load_rather_than_instances(args.instances_dir, args.corpus_name))
    by_paper = defaultdict(list)
    for r in instances:
        if r["doc_id"] in qualifying:
            by_paper[r["doc_id"]].append(r)

    rng = random.Random(args.seed)
    papers = list(by_paper)
    rng.shuffle(papers)
    for plist in by_paper.values():
        rng.shuffle(plist)

    picked = []
    per_paper_taken = defaultdict(int)
    # round-robin across papers so the batch isn't dominated by one outlier
    changed = True
    while len(picked) < args.count and changed:
        changed = False
        for doc in papers:
            if len(picked) >= args.count:
                break
            if per_paper_taken[doc] >= args.max_per_paper:
                continue
            for r in by_paper[doc]:
                key = instance_key(r["doc_id"], r["sentence"], r["local_start"])
                if key in used_keys:
                    continue
                used_keys.add(key)
                picked.append(r)
                per_paper_taken[doc] += 1
                changed = True
                break

    if len(picked) < args.count:
        print(f"WARNING: only found {len(picked)}/{args.count} eligible instances "
              f"(pool exhausted at max-per-paper={args.max_per_paper})", file=sys.stderr)

    n_papers_used = len(per_paper_taken)
    print(f"picked {len(picked)} instances from {n_papers_used} distinct papers "
          f"(mean {len(picked)/max(n_papers_used,1):.1f}/paper)", file=sys.stderr)

    corpus_label = f"high_count_{args.corpus_name}"
    next_num = max_num + 1
    with open(pub_path, "a") as pub, open(prov_path, "a") as prov_out:
        for i, rec in enumerate(picked):
            item_id = f"item_{next_num + i}"
            pub.write(json.dumps({
                "item_id": item_id, "sentence": rec["sentence"],
                "local_start": rec["local_start"], "local_end": rec["local_end"],
            }) + "\n")
            prov_out.write(json.dumps({
                "item_id": item_id, "corpus": corpus_label, "doc_id": rec["doc_id"],
                "pattern_id": rec["pattern_id"], "char_start": rec["char_start"],
                "char_end": rec["char_end"], "match_text": rec["match_text"],
                "paper_total_count": counts[rec["doc_id"]],
            }) + "\n")

    print(f"DONE appended {len(picked)} items -> {pub_path} "
          f"(total now {len(existing_pub) + len(picked)})", file=sys.stderr)


if __name__ == "__main__":
    main()

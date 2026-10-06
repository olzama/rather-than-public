#!/usr/bin/env python3
r"""
Paper-level encounter model: how likely is a reader to meet at least one
"rather than" that annoys them, in a paper with k uses?

For a reader whose per-use annoying rate in a corpus is p, the chance of
at least one annoying use in a paper with k uses is 1 - (1 - p)^k,
assuming uses are judged independently at a constant rate. The script
averages this over all papers of each corpus (papers without the phrase
count as 0), and reports the expected number of annoying uses per paper
(p * k). p comes from each annotator's labels on the random strata of
that corpus (garbled excluded): ACL 2019 for ACL 2019; dataset (2), v1
and latest for arXiv 2026. Where p is 0, the upper end of its 95% Wilson
interval is also reported, as a ceiling.

"either" is annoying for an item if either annotator of --pair labeled it
annoying (items both labeled, garbled by neither).

Usage:
    python3 encounter_model.py ../data/annotation ../data/antithesis_instances \
        --papers acl2019=4863 arxiv2026=1841 [--pair A1 A2] [--batch 1] \
        [--exclude ../data/excluded_documents.tsv]

--papers gives each corpus's full size; documents listed for that corpus
in --exclude (language_filter.py output) are removed from both the
paper count and the use counts.
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

from multi_annotator import GROUPS, wilson

CORPUS_GROUP = {"acl2019": "ACL 2019", "arxiv2026": "arXiv 2026 random"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("instances_dir")
    ap.add_argument("--papers", nargs="+", required=True, help="corpus=number_of_papers")
    ap.add_argument("--pair", nargs=2, default=["A1", "A2"])
    ap.add_argument("--batch", type=int, help="only items of this annotation batch (default: all)")
    ap.add_argument("--exclude", help="TSV from language_filter.py")
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    n_papers = {k: int(v) for k, v in (s.split("=") for s in args.papers)}
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl")) if "excluded" not in r}
    pool = {r["item_id"] for r in map(json.loads, open(d / "items_public.jsonl"))
            if args.batch is None or r.get("batch", 1) == args.batch}
    pool &= set(prov)
    labels = collections.defaultdict(dict)
    for l in open(d / "human_labels.jsonl"):
        r = json.loads(l)
        if r["item_id"] in pool:
            labels[r["annotator"]][r["item_id"]] = r["label"]
    a, b = args.pair
    labels["either"] = {i: ("annoying" if "annoying" in (labels[a][i], labels[b][i]) else "other")
                        for i in labels[a] if i in labels[b] and "garbled" not in (labels[a][i], labels[b][i])}

    def p_of(name, group):
        vals = [v for i, v in labels[name].items() if GROUPS[prov[i]["corpus"]] == group and v != "garbled"]
        k = sum(v == "annoying" for v in vals)
        return k, len(vals)

    skip = collections.defaultdict(set)
    if args.exclude:
        for line in list(open(args.exclude))[1:]:
            c, doc = line.split("\t")[:2]
            skip[c].add(doc)
    for corpus, total in n_papers.items():
        total -= len(skip[corpus])
        counts = collections.Counter(json.loads(l)["doc_id"] for l in open(Path(args.instances_dir) / f"{corpus}.jsonl")
                                     if json.loads(l)["pattern_id"] == "rather_than" and json.loads(l)["doc_id"] not in skip[corpus])
        ks = list(counts.values()) + [0] * (total - len(counts))
        print(f"{corpus}: {total} papers, {len(counts)} with 'rather than' ({len(counts) / total:.0%}); "
              f"uses per paper mean {statistics.mean(ks):.2f}, median {statistics.median(ks):g}, max {max(ks)}")
        print(f"  {'reader':8s} {'p':>18s} {'P(>=1 annoying)':>16s} {'annoying uses/paper':>20s}")
        for name in (a, b, "either"):
            k, n = p_of(name, CORPUS_GROUP[corpus])
            ps = [("", k / n)] + ([("ceiling", wilson(k, n)[1])] if k == 0 else [])
            for tag, p in ps:
                reach = statistics.mean(1 - (1 - p) ** x for x in ks)
                print(f"  {name:8s} {f'{p:.3f} ({k}/{n}) {tag}':>18s} {reach:16.1%} {p * statistics.mean(ks):20.2f}")
        print()


if __name__ == "__main__":
    main()

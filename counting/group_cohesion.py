#!/usr/bin/env python3
r"""
Are annoying "rather than" items semantically similar to each other?

For each unit -- X, Y (from extract_xy.py), the sentence, and the paragraph
(the sentence with --para-k sentences on each side, embedded as one text) --
over the arXiv 2026 items labeled annoying or legitimate:

  cohesion   mean pairwise cosine among annoying items, compared with the
             same statistic for random subsets of the same size drawn from
             all items (permutation p: share of random subsets at least as
             cohesive). Mean pairwise cosine among legitimate items and
             between the classes is reported for reference.
  kNN        for each annoying item, the share of its --knn nearest
             neighbours (among all items) that are annoying, averaged;
             compared with the same statistic under shuffled labels.

With --exclude-same-paper, pairs of items from the same paper (any arXiv
version; for LLM-written papers, the same source paper) are ignored in both tests, so shared topic within a paper cannot
drive the result.

--corpora selects the items' corpora (default: the arXiv 2026 strata); for
LLM-written items (corpus llm_<model>, e.g. annotation batch 3), paragraph
context is read from --llm-text-dir/<model>/<doc_id>.txt (llm_instances.py).

Embeddings (OpenAI, default text-embedding-3-large) are cached in --cache.

Usage:
    python3 group_cohesion.py ../data/annotation ../../data \
        --xy ../data/annotation/xy_alternatives.jsonl --key-file ../../LYS-API-key.txt \
        [--cache ...] [--para-k 2] [--knn 5] [--perms 10000] [--exclude-same-paper] [--annotator A1]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from semantic_similarity import ARXIV2026, TEXT_DIRS, Embedder, context_sentences, norm


def mean_pairwise(S, idx, allowed):
    """Mean similarity over pairs of distinct items in idx that `allowed` permits."""
    sub, ok = S[np.ix_(idx, idx)], allowed[np.ix_(idx, idx)]
    return sub[ok].mean()


def nearest(S, allowed, k):
    return np.argsort(-np.where(allowed, S, -np.inf), axis=1)[:, :k]


def knn_share(nn, is_ann):
    return is_ann[nn[np.where(is_ann)[0]]].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("data_dir")
    ap.add_argument("--xy", required=True)
    ap.add_argument("--corpora", nargs="+", default=sorted(ARXIV2026))
    ap.add_argument("--llm-text-dir")
    ap.add_argument("--annotator", default="A1")
    ap.add_argument("--para-k", type=int, default=2)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exclude-same-paper", action="store_true",
                    help="ignore pairs of items from the same paper (any version)")
    ap.add_argument("--model", default="text-embedding-3-large")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--key-file")
    args = ap.parse_args()
    d, data = Path(args.annotation_dir), Path(args.data_dir)
    key = os.environ.get("OPENAI_API_KEY") or (Path(args.key_file).read_text().strip() if args.key_file else None)
    if not key:
        sys.exit("no API key: set OPENAI_API_KEY or pass --key-file")
    emb = Embedder(key, args.model, args.cache or d / "embeddings_cache.jsonl")

    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl")) if "excluded" not in r}
    pub = {json.loads(l)["item_id"]: json.loads(l) for l in open(d / "items_public.jsonl")}
    pub = {i: v for i, v in pub.items() if i in prov}
    xy = {json.loads(l)["item_id"]: json.loads(l) for l in open(args.xy)}
    labels = {}
    for l in open(d / "human_labels.jsonl"):
        r = json.loads(l)
        if r["annotator"] == args.annotator:
            labels[r["item_id"]] = r["label"]
    items = sorted(i for i in pub if prov[i]["corpus"] in args.corpora and "repeat_of" not in prov[i]
                   and labels.get(i) in ("annoying", "legitimate") and i in xy)

    texts, units = {}, {"X": [], "Y": [], "sentence": [], "paragraph": []}
    for i in items:
        p = prov[i]
        path = (Path(args.llm_text_dir) / p["corpus"][4:] / f"{p['doc_id']}.txt" if p["corpus"].startswith("llm_")
                else data / TEXT_DIRS[p["corpus"]] / f"{p['doc_id']}.txt")
        if path not in texts:
            texts[path] = path.read_text(errors="replace")
        before, after = context_sentences(texts[path], p["char_start"], p["char_end"], args.para_k)
        units["X"].append(xy[i]["x"])
        units["Y"].append(xy[i]["y"])
        units["sentence"].append(pub[i]["sentence"])
        units["paragraph"].append(" ".join(norm(s) for s in before + [pub[i]["sentence"]] + after))

    is_ann = np.array([labels[i] == "annoying" for i in items])
    # same paper: any arXiv version; for LLM-written papers, the source paper (any model or stage)
    paper = np.array([re.sub(r"_v\d+$", "", prov[i]["doc_id"]).split("__")[-1] for i in items])
    allowed = ~np.eye(len(items), dtype=bool)
    if args.exclude_same_paper:
        allowed &= paper[:, None] != paper[None, :]
    ann, leg = np.where(is_ann)[0], np.where(~is_ann)[0]
    rng = np.random.default_rng(args.seed)
    print(f"arXiv 2026 items: {len(ann)} annoying, {len(leg)} legitimate; embeddings: {args.model}; "
          f"paragraph = sentence +/- {args.para_k}; {args.perms} permutations"
          f"{'; same-paper pairs excluded' if args.exclude_same_paper else ''}\n")
    print(f"{'unit':10s} {'ann-ann':>8s} {'leg-leg':>8s} {'ann-leg':>8s} {'rand-41':>8s} {'p_coh':>7s}"
          f" {'kNN ann':>8s} {'kNN null':>9s} {'p_knn':>7s}")
    for unit, strs in units.items():
        V = emb.embed(strs)
        S = V @ V.T
        aa, ll = mean_pairwise(S, ann, allowed), mean_pairwise(S, leg, allowed)
        al = S[np.ix_(ann, leg)][allowed[np.ix_(ann, leg)]].mean()
        null = np.array([mean_pairwise(S, rng.choice(len(items), len(ann), replace=False), allowed)
                         for _ in range(args.perms)])
        p_coh = (1 + (null >= aa).sum()) / (1 + args.perms)
        nn = nearest(S, allowed, args.knn)
        kn = knn_share(nn, is_ann)
        kperm = min(args.perms, 2000)
        knull = np.array([knn_share(nn, rng.permutation(is_ann)) for _ in range(kperm)])
        p_knn = (1 + (knull >= kn).sum()) / (1 + kperm)
        print(f"{unit:10s} {aa:8.3f} {ll:8.3f} {al:8.3f} {null.mean():8.3f} {p_coh:7.3f}"
              f" {kn:8.3f} {knull.mean():9.3f} {p_knn:7.3f}")


if __name__ == "__main__":
    main()

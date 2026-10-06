#!/usr/bin/env python3
r"""
Score each "rather than" item by how close its rejected alternative (Y) is
to clear straw men, and validate the score against human judgments.

Seeds (fixed before scoring): the Ys that both straw-man coders coded as
straw men (AGREED_ITEMS), and the invented examples in the straw-man
definition shown to coders (DEFINITION_SEEDS), which come from outside the
data. Score = maximum cosine similarity (text-embedding-3-large) of an
item's Y to the seeds. Seed items themselves are left out of the
validation.

With --unit clause, items and seeds are the clause "X rather than Y" and
the seeds are the agreed straw men only (the definition examples have no X).

Validation, on pool items with an LLM-extracted Y (xy_alternatives.jsonl):
AUC of the score for each coder's straw-man codes (coded items), and for
each annotator's annoyance labels and the union (arXiv 2026 items); plus
the top-ranked items, for reading.

Usage:
    python3 egregious_similarity.py ../data/annotation --xy ../data/annotation/xy_alternatives.jsonl \
        --key-file ../../LYS-API-key.txt [--cache ...] [--top 25] [--out scores.jsonl]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats

from semantic_similarity import Embedder

AGREED_ITEMS = ["item_1764", "item_1792", "item_1809", "item_1856", "item_1880", "item_2037"]
DEFINITION_SEEDS = ["guaranteeing correct output on every input", "proving the method works for all languages",
                    "tuning hyperparameters on the test set", "reporting only the best of many runs",
                    "picking answers at random", "ignoring the input altogether"]
ARXIV = {"arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"}


def auc(scores, labels):
    """Mann-Whitney AUC (ties count half), from ranks."""
    s = np.asarray(scores, float); l = np.asarray(labels, bool)
    npos, nneg = int(l.sum()), int((~l).sum())
    if not npos or not nneg:
        return float("nan"), npos, nneg
    r = stats.rankdata(s)
    return (r[l].sum() - npos * (npos + 1) / 2) / (npos * nneg), npos, nneg


def perm_p(scores, labels, obs, n=5000, seed=1):
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    s = np.array(scores)
    hits = 0
    for _ in range(n):
        rng.shuffle(labels)
        if auc(s, labels)[0] >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--xy", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--model", default="text-embedding-3-large")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out")
    ap.add_argument("--unit", choices=["y", "clause"], default="y")
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    key = open(args.key_file).read().strip()
    emb = Embedder(key, args.model, args.cache or d / "embeddings_cache.jsonl")
    xy = {r["item_id"]: r for r in map(json.loads, open(args.xy)) if r.get("y")}
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl"))
            if "repeat_of" not in r and "excluded" not in r}
    pub = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl"))}
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        labels[r["annotator"]][r["item_id"]] = r["label"]
    codes = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "strawman" / "human_codes.jsonl")):
        codes[r["annotator"]][r["item_id"]] = r["code"]

    unit = (lambda i: xy[i]["y"]) if args.unit == "y" else (lambda i: f"{xy[i]['x']} rather than {xy[i]['y']}")
    seeds = [unit(i) for i in AGREED_ITEMS] + (DEFINITION_SEEDS if args.unit == "y" else [])
    items = sorted(i for i in xy if i in prov and i in pub and i not in AGREED_ITEMS and xy[i].get("x"))
    S = emb.embed(seeds)
    Y = emb.embed([unit(i) for i in items])
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    sim = Y @ S.T
    score = {i: float(v) for i, v in zip(items, sim.max(axis=1))}
    nearest = {i: seeds[int(k)] for i, k in zip(items, sim.argmax(axis=1))}
    print(f"seeds: {len(seeds)} ({len(AGREED_ITEMS)} agreed straw men, {len(DEFINITION_SEEDS)} definition examples); "
          f"scored items: {len(items)}")

    print("\nvalidation (AUC; permutation p, one-sided):")
    for coder in ("A1", "A2"):
        ids = [i for i in items if codes[coder].get(i) in ("strawman", "not_strawman")]
        a, npos, nneg = auc([score[i] for i in ids], [codes[coder][i] == "strawman" for i in ids])
        p = perm_p([score[i] for i in ids], [codes[coder][i] == "strawman" for i in ids], a)
        print(f"  {coder}'s straw-man codes: AUC {a:.2f} ({npos} straw men, {nneg} not), p = {p:.3g}")
    arx = [i for i in items if prov[i]["corpus"] in ARXIV]
    union = {i: "annoying" in (labels["A1"].get(i), labels["A2"].get(i)) for i in arx}
    for name, lab in (("A1", None), ("A2", None), ("Either", union)):
        if lab is None:
            ids = [i for i in arx if labels[name].get(i) in ("annoying", "legitimate")]
            y = [labels[name][i] == "annoying" for i in ids]
        else:
            ids = [i for i in arx if labels["A1"].get(i) not in (None, "garbled") and labels["A2"].get(i) not in (None, "garbled")]
            y = [lab[i] for i in ids]
        a, npos, nneg = auc([score[i] for i in ids], y)
        p = perm_p([score[i] for i in ids], y, a)
        print(f"  {name} annoyance (arXiv 2026): AUC {a:.2f} ({npos} annoying, {nneg} other), p = {p:.3g}")

    print(f"\ntop {args.top} items by score:")
    for i in sorted(items, key=lambda i: -score[i])[:args.top]:
        print(f"  {score[i]:.3f} {i} {prov[i]['corpus']:22s} {unit(i)[:70]!r:74s} ~ {nearest[i][:40]!r} "
              f"| ann {labels['A1'].get(i, '-')[:3]}/{labels['A2'].get(i, '-')[:3]} "
              f"| sm {codes['A1'].get(i, '-')[:3]}/{codes['A2'].get(i, '-')[:3]}")
    if args.out:
        with open(args.out, "w") as f:
            for i in items:
                f.write(json.dumps({"item_id": i, "score": score[i], "nearest_seed": nearest[i]}) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Embedding-based semantic measures for annoying vs. legitimate "rather than"
items (arXiv 2026 strata; ACL 2019 has no annoying items). For each item:

  xy          cos(X, Y): how similar the affirmed and rejected alternatives are
  y_ctx_K     max cos(Y, s) over the K sentences before and K after: whether
              the rejected alternative is already present nearby
  x_ctx_K     the same for X
  sent_ctx_K  mean cos(sentence, s) over those 2K sentences: how much the
              sentence repeats its surroundings
  typicality  mean cos(sentence, every other arXiv 2026 pool sentence): how
              formulaic the sentence is

X and Y come from extract_xy.py. Context sentences are taken from the full
text with the same sentence splitter used to extract the instances.
Embeddings (OpenAI, default text-embedding-3-large) are cached in
--cache, keyed by model and text, so reruns cost nothing.

For each measure the report gives the median per class, the Mann-Whitney p,
the AUC for annoying (0.5 = no difference), and Benjamini-Hochberg q-values
across measures.

Usage:
    python3 semantic_similarity.py ../data/annotation ../../data \
        --xy ../data/annotation/xy_alternatives.jsonl --key-file ../../LYS-API-key.txt \
        [--cache ../data/annotation/embeddings_cache.jsonl] [--k 1 3] [--annotator A1] [--out items.jsonl]
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data_collection"))
import llm_judge as J  # noqa: E402
from extract_antithesis_instances import sentence_spans, sentence_for_span  # noqa: E402

ARXIV2026 = {"arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"}
TEXT_DIRS = {
    "arxiv2026": "arxiv2026/text", "high_count_arxiv2026": "arxiv2026/text",
    "arxiv_v1": "arxiv_versions/text", "arxiv_latest": "arxiv_versions/text",
    "acl2019": "acl2019/text",
}


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


class Embedder:
    def __init__(self, key, model, cache_path):
        self.key, self.model, self.cache_path = key, model, Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            for l in open(self.cache_path):
                r = json.loads(l)
                self.cache[r["h"]] = r["v"]

    def h(self, text):
        return hashlib.sha1(f"{self.model}\n{text}".encode()).hexdigest()

    def embed(self, texts):
        texts = [norm(t) or "." for t in texts]
        missing = sorted({t for t in texts if self.h(t) not in self.cache})
        with open(self.cache_path, "a") as f:
            for i in range(0, len(missing), 256):
                batch = missing[i:i + 256]
                resp = J.post(self.key, {"model": self.model, "input": batch}, endpoint="embeddings")
                for t, d in zip(batch, resp["data"]):
                    v = [round(x, 6) for x in d["embedding"]]
                    self.cache[self.h(t)] = v
                    f.write(json.dumps({"h": self.h(t), "v": v}) + "\n")
        out = np.array([self.cache[self.h(t)] for t in texts], dtype=float)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def context_sentences(text, char_start, char_end, k):
    spans = sentence_spans(text)
    _, _, first, last = sentence_for_span(spans, char_start, char_end)
    if first is None:
        return [], []
    before = [text[a:b] for a, b in spans[max(0, first - k):first]]
    after = [text[a:b] for a, b in spans[last + 1:last + 1 + k]]
    return [s for s in before if norm(s)], [s for s in after if norm(s)]


def bh(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    q, prev = [0.0] * len(pvals), 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, pvals[i] * len(pvals) / rank)
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("data_dir", help="root of the corpus text directories (arxiv2026/text, ...)")
    ap.add_argument("--xy", required=True)
    ap.add_argument("--annotator", default="A1")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--model", default="text-embedding-3-large")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--key-file")
    ap.add_argument("--out", help="write per-item measures (jsonl) here")
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
    items = [i for i in pub if prov[i]["corpus"] in ARXIV2026 and "repeat_of" not in prov[i]
             and labels.get(i) in ("annoying", "legitimate") and i in xy]

    kmax = max(args.k)
    texts = {}
    rows = []
    for i in items:
        p = prov[i]
        path = data / TEXT_DIRS[p["corpus"]] / f"{p['doc_id']}.txt"
        if path not in texts:
            texts[path] = path.read_text(errors="replace")
        before, after = context_sentences(texts[path], p["char_start"], p["char_end"], kmax)
        rows.append({"item_id": i, "label": labels[i], "sentence": pub[i]["sentence"],
                     "x": xy[i]["x"], "y": xy[i]["y"], "before": before, "after": after})

    all_texts = []
    for r in rows:
        all_texts += [r["sentence"], r["x"], r["y"]] + r["before"] + r["after"]
    emb.embed(all_texts)  # one pass fills the cache

    sent_vecs = emb.embed([r["sentence"] for r in rows])
    for n, r in enumerate(rows):
        s, x, y = sent_vecs[n], *emb.embed([r["x"], r["y"]])
        r["xy"] = float(x @ y)
        for k in args.k:
            ctx = r["before"][-k:] + r["after"][:k]
            if not ctx:
                continue
            c = emb.embed(ctx)
            r[f"y_ctx_{k}"] = float((c @ y).max())
            r[f"x_ctx_{k}"] = float((c @ x).max())
            r[f"sent_ctx_{k}"] = float((c @ s).mean())
        others = np.delete(sent_vecs, n, axis=0)
        r["typicality"] = float((others @ s).mean())

    measures = ["xy"] + [f"{m}_{k}" for k in args.k for m in ("y_ctx", "x_ctx", "sent_ctx")] + ["typicality"]
    res = []
    for m in measures:
        a = [r[m] for r in rows if r["label"] == "annoying" and m in r]
        b = [r[m] for r in rows if r["label"] == "legitimate" and m in r]
        u = stats.mannwhitneyu(a, b)
        res.append((m, len(a), len(b), statistics.median(a), statistics.median(b), u.pvalue, u.statistic / (len(a) * len(b))))
    qs = bh([r[5] for r in res])
    print(f"arXiv 2026 items: {sum(r['label'] == 'annoying' for r in rows)} annoying, "
          f"{sum(r['label'] == 'legitimate' for r in rows)} legitimate; embeddings: {args.model}\n")
    print(f"{'measure':14s} {'n_ann':>6s} {'n_leg':>6s} {'med_ann':>8s} {'med_leg':>8s} {'AUC':>6s} {'p':>7s} {'q':>7s}")
    for (m, na, nb, ma, mb, p, a), q in zip(res, qs):
        print(f"{m:14s} {na:6d} {nb:6d} {ma:8.3f} {mb:8.3f} {a:6.2f} {p:7.3f} {q:7.3f}")
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("before", "after")}) + "\n")


if __name__ == "__main__":
    main()

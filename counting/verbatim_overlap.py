#!/usr/bin/env python3
r"""
Do LLM-written papers copy text verbatim from the paper they were written
from? The model saw only the title and abstract, so text shared with the
rest of the original suggests memorization.

For each model, stage (generated, revised) and source paper: the share of
the LLM text's word n-grams (lowercased \w+ tokens; n = 8 and 13) that occur
in the original body text but not in its abstract, and the longest shared
run of words not contained in the abstract. The same is computed against a
different original paper from the same corpus (a chance baseline for
formulaic phrasing). Also counts "rather than" sentences in the LLM text that
share a 13-gram containing "rather than" with the original.

Usage:
    python3 verbatim_overlap.py ../data/generated/gpt_papers/batch2 ../data/generated/gpt_papers/batch3 --data-dir ../data \
        [--models gpt-4o gpt-5.6-sol gpt-6-sol] [--examples 5]
"""
import argparse
import gzip
import json
import random
import re
import statistics
from pathlib import Path

WORD = re.compile(r"\w+")


def toks(text):
    return [w.lower() for w in WORD.findall(text)]


def ngrams(t, n):
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def longest_run(gen, ref8, abs8):
    """Longest run of words in gen covered by consecutive 8-grams that occur in the
    reference and not in its abstract (so copying the given abstract never counts).
    Returns (length in words, the run)."""
    best, best_i, cur = 0, 0, 0
    for i in range(len(gen) - 7):
        q = tuple(gen[i:i + 8])
        cur = cur + 1 if q in ref8 and q not in abs8 else 0
        if cur and cur + 7 > best:
            best, best_i = cur + 7, i - cur + 1
    return best, " ".join(gen[best_i:best_i + best])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generation_dirs", nargs="+", help="one or more generation runs (data/generated/gpt_papers/batch*)")
    ap.add_argument("--data-dir", required=True, help="the repository's data folder (acl2019/, arxiv2026/ text)")
    ap.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-5.6-sol", "gpt-6-sol"])
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()
    from generation_view import combined
    gdir, ddir = combined(args.generation_dirs, args.models), Path(args.data_dir)
    ids = [l.rstrip("\n").split("\t") for l in open(gdir / args.models[0] / "sample_ids.tsv") if not l.startswith("#")]
    abstracts = {}
    for corpus in ("acl2019", "arxiv2026"):
        for l in gzip.open(ddir / corpus / "sections.jsonl.gz", "rt"):
            r = json.loads(l)
            abstracts[r["doc_id"]] = r.get("abstract") or next((s["text"] for s in r.get("sections", []) if s.get("category") == "abstract"), "")
    orig = {}
    for corpus, doc in ids:
        t = toks((ddir / corpus / "text_body" / f"{doc}.txt").read_text(errors="ignore"))
        a = toks(abstracts.get(doc, ""))
        orig[doc] = {"corpus": corpus, "t": t, "s": " " + " ".join(t) + " ", "a": " " + " ".join(a) + " ",
                     "a8": ngrams(a, 8), "a13": ngrams(a, 13), "g8": ngrams(t, 8), "g13": ngrams(t, 13)}
    rng = random.Random(1)
    other = {}
    for corpus in ("acl2019", "arxiv2026"):
        docs = [d for c, d in ids if c == corpus]
        shuffled = docs[1:] + docs[:1]
        other.update(dict(zip(docs, shuffled)))
    rt = re.compile(r"[^.!?]*\brather than\b[^.!?]*[.!?]", re.I)
    examples = []
    print(f"{'model':12s} {'stage':9s} {'source':9s} {'8-gram %':>9s} {'(other)':>8s} {'13-gram %':>10s} {'(other)':>8s} "
          f"{'longest':>8s} {'(other)':>8s} {'RT copied':>10s}")
    for m in args.models:
        for stage in ("generated", "revised"):
            by = {}
            for corpus, doc in ids:
                f = gdir / m / f"{stage}_papers" / f"{doc}.md"
                if not f.exists():
                    continue
                text = f.read_text(errors="ignore")
                g = toks(text)
                o, x = orig[doc], orig[other[doc]]
                row = {}
                for n in (8, 13):
                    grams = ngrams(g, n)
                    if not grams:
                        continue
                    own = [q for q in grams if q in o[f"g{n}"] and q not in o[f"a{n}"]]
                    oth = [q for q in grams if q in x[f"g{n}"] and q not in x[f"a{n}"]]
                    row[n] = (100 * len(own) / len(grams), 100 * len(oth) / len(grams))
                lr, run = longest_run(g, o["g8"], o["a8"])
                lx, _ = longest_run(g, x["g8"], x["a8"])
                copied = 0
                for sent in rt.findall(text):
                    st = toks(sent)
                    k = [i for i in range(len(st) - 1) if st[i] == "rather" and st[i + 1] == "than"]
                    if any(tuple(st[i:i + 13]) in o["g13"] and tuple(st[i:i + 13]) not in o["a13"]
                           for kk in k for i in range(max(0, kk - 11), kk + 1) if len(st[i:i + 13]) == 13):
                        copied += 1
                by.setdefault(corpus, []).append((row, lr, lx, copied, len(rt.findall(text))))
                if lr >= 20:
                    examples.append((lr, m, stage, doc, run))
            for corpus, rows in sorted(by.items()):
                mean = lambda f: statistics.mean(f(r) for r in rows)
                print(f"{m:12s} {stage:9s} {corpus:9s} {mean(lambda r: r[0][8][0]):9.2f} {mean(lambda r: r[0][8][1]):8.2f} "
                      f"{mean(lambda r: r[0][13][0]):10.2f} {mean(lambda r: r[0][13][1]):8.2f} "
                      f"{statistics.median(r[1] for r in rows):8.0f} {statistics.median(r[2] for r in rows):8.0f} "
                      f"{sum(r[3] for r in rows):5d}/{sum(r[4] for r in rows):<5d}")
    print(f"\n(longest: median, words; RT copied: 'rather than' sentences sharing a 13-gram around the phrase with the original / all)")
    print(f"\nlongest verbatim runs outside the abstract (>= 20 words): {len(examples)}")
    for lr, m, stage, doc, run in sorted(examples, reverse=True)[:args.examples]:
        print(f"  {lr:3d} words  {m} {stage} {doc}: {run[:220]}")


if __name__ == "__main__":
    main()

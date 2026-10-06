#!/usr/bin/env python3
r"""
Blind human spot check of inverted items (the LLM rates the rejected
alternative Y above the affirmed X). Draws N items the LLM rated inverted
(x < y) and N it did not (x >= y) from the arXiv items of the stance analysis,
shuffles them, and writes the key and the data for the spot-check page. The
page asks which alternative the author presents more favorably (X, Y or
neither) and shows no LLM rating or annoyance label.

Usage:
    python3 stance_spot_check.py ../data/annotation ../data/annotation/stance/spot_check [--n 40]
Scoring: ../counting/stance_spot_check_agreement.py
"""
import argparse
import json
import random
import re
from pathlib import Path

ARXIV = ["arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"]


def find(s, phrase):
    m = re.search(r"\s+".join(re.escape(w) for w in phrase.split()), s)
    return (m.start(), m.end()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    load = lambda f: {r["item_id"]: r for r in map(json.loads, open(d / f))}
    pub, prov, xy = load("items_public.jsonl"), load("items_provenance.jsonl"), load("xy_alternatives.jsonl")
    sc = load("stance/stance__gpt-6-sol.jsonl")

    rows = {}
    for i, s in sc.items():
        if prov[i]["corpus"] not in ARXIV or not xy.get(i, {}).get("valid"):
            continue
        sent = " ".join(pub[i]["sentence"].split())
        spx, spy = find(sent, xy[i]["x"]), find(sent, xy[i]["y"])
        if not spx or not spy or spx[0] < spy[1] and spy[0] < spx[1]:
            continue
        rows[i] = {"sent": sent, "x": spx, "y": spy, "inverted": s["x"] < s["y"]}
    rng = random.Random(args.seed)
    inv = sorted(i for i in rows if rows[i]["inverted"])
    non = sorted(i for i in rows if not rows[i]["inverted"])
    picked = rng.sample(inv, args.n) + rng.sample(non, args.n)
    rng.shuffle(picked)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    with open(out / "key.jsonl", "w") as k:
        for n, i in enumerate(picked, 1):
            r = rows[i]
            k.write(json.dumps({"n": n, "item_id": i, "llm_x": sc[i]["x"], "llm_y": sc[i]["y"],
                                "inverted": r["inverted"]}) + "\n")
            (a, b), (c, e) = sorted([r["x"], r["y"]])
            first = "x" if r["x"][0] < r["y"][0] else "y"
            s = r["sent"]
            items.append({"n": n, "b": " ".join(pub[i]["context_before"].split()),
                          "a": " ".join(pub[i]["context_after"].split()),
                          "parts": [[s[:a], ""], [s[a:b], first], [s[b:c], ""],
                                    [s[c:e], "y" if first == "x" else "x"], [s[e:], ""]]})
    json.dump({"items": items}, open(out / "page_data.json", "w"), ensure_ascii=False)
    print(f"{len(inv)} inverted, {len(non)} non-inverted candidates; wrote {len(picked)} items to {out}")


if __name__ == "__main__":
    main()

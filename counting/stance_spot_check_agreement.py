#!/usr/bin/env python3
r"""
Score the human stance spot check (data_collection/stance_spot_check.py)
against the LLM ratings: how often the items the LLM rated inverted (x < y)
are judged by the human to favor Y, and the same for the items it did not.

Answers come from the spot-check web page: one {"n", "answer", "note"} per
item, answer in {"x", "y", "neither"}.

Usage:
    python3 stance_spot_check_agreement.py ../data/annotation/stance/spot_check/answers.jsonl \
        ../data/annotation/stance/spot_check/key.jsonl
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("key")
    args = ap.parse_args()
    key = {k["n"]: k for k in map(json.loads, open(args.key))}
    ans = {r["n"]: r["answer"] for r in map(json.loads, open(args.answers)) if r.get("answer")}
    table = collections.Counter((key[n]["inverted"], a) for n, a in ans.items())
    print(f"{len(ans)} of {len(key)} items answered")
    for inv in (True, False):
        n = sum(table[inv, a] for a in ("x", "y", "neither"))
        row = "  ".join(f"{a} {table[inv, a]}" for a in ("x", "y", "neither"))
        print(f"LLM {'inverted    ' if inv else 'not inverted'} (n={n}): human {row}")
    tp, fp = table[True, "y"], sum(table[True, a] for a in ("x", "neither"))
    fn = table[False, "y"]
    if tp + fp:
        print(f"inverted items the human judges to favor Y: {tp}/{tp + fp} ({tp / (tp + fp):.0%})")
    if fn + tp:
        print(f"items the human judges to favor Y that the LLM rated inverted: {tp}/{tp + fn}")
    pairs = [(key[n]["inverted"], a == "y") for n, a in ans.items()]
    if pairs:
        po = sum(a == b for a, b in pairs) / len(pairs)
        pa = sum(a for a, _ in pairs) / len(pairs)
        pb = sum(b for _, b in pairs) / len(pairs)
        pe = pa * pb + (1 - pa) * (1 - pb)
        print(f"agreement (inverted vs favors Y): {po:.0%}, kappa {(po - pe) / (1 - pe):.2f}")


if __name__ == "__main__":
    main()

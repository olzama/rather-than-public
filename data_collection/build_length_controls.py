#!/usr/bin/env python3
r"""
Length-matched control for the reward-model test. For each minimal pair
(build_reward_pairs.py) the "rather than Y" clause is replaced by a
non-contrastive clause of (about) the same length, written separately for each
item by Claude Opus 5.5 subagents (replacements.jsonl: item_id, replacement). The control sentence has the
same slot, length and position as the original but no antithesis; comparing
original vs. control separates the effect of the antithesis from the effect of
simply having more text.

Output: pairs_control.jsonl, in the format of pairs.jsonl with "original" =
the control sentence, "deleted" unchanged, plus "true_original" and
"replacement". Score it with reward_scores.py like pairs.jsonl. Items whose
replacement fails validation (clause not found in the sentence, a contrast
marker in the replacement, word count off by more than --tol) are reported and
left out.

Usage:
    python3 build_length_controls.py ../data/reward_pairs/pairs.jsonl replacements.jsonl \
        ../data/reward_pairs/pairs_control.jsonl [--tol 1]
"""
import argparse
import json
import re

CONTRAST = re.compile(r"\b(rather|instead|unlike|whereas|than|not|no|never|without|versus|vs|opposed|"
                      r"contrast|but|however|n't)\b|n't", re.I)


def substitute(text, clause, replacement):
    """Replace the clause in text (matching any whitespace between its words); None unless found once."""
    pat = r"\s+".join(re.escape(w) for w in clause.split())
    hits = list(re.finditer(pat, text))
    if len(hits) != 1:
        return None
    m = hits[0]
    return text[:m.start()] + replacement + text[m.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("replacements")
    ap.add_argument("out")
    ap.add_argument("--tol", type=int, default=1, help="allowed word-count difference")
    args = ap.parse_args()
    rep = {r["item_id"]: r["replacement"].strip() for r in map(json.loads, open(args.replacements))}
    n = 0
    bad = []
    with open(args.out, "w") as f:
        for p in map(json.loads, open(args.pairs)):
            clause = " ".join(p["clause"].split()).lstrip(" ,")  # a leading comma stays in the sentence
            r = rep.get(p["item_id"])
            if r is not None:
                r = r.lstrip(" ,")
            if r is None:
                bad.append((p["item_id"], "no replacement"))
                continue
            if CONTRAST.search(r):
                bad.append((p["item_id"], f"contrast marker: {r!r}"))
                continue
            if abs(len(r.split()) - len(clause.split())) > args.tol:
                bad.append((p["item_id"], f"length {len(r.split())} vs {len(clause.split())}: {r!r}"))
                continue
            ctl = substitute(p["original"], clause, r)
            if ctl is None:
                bad.append((p["item_id"], "clause not found exactly once"))
                continue
            f.write(json.dumps({**p, "original": ctl, "true_original": p["original"], "replacement": r}) + "\n")
            n += 1
    print(f"wrote {n} controls to {args.out}; rejected {len(bad)}")
    for i, why in bad:
        print(f"  {i}: {why}")


if __name__ == "__main__":
    main()

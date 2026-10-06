#!/usr/bin/env python3
r"""
Which aspect of a response's rating goes with an antithesis pattern? Uses the
raw UltraFeedback release (openbmb/UltraFeedback: per prompt, four
completions, each rated 1-5 by GPT-4 on helpfulness, honesty, instruction
following and truthfulness). Specified before running: the prediction is
that "rather than" raises the honesty rating more than the other aspects.

For each aspect and pattern: a within-prompt regression (prompt fixed
effects, by demeaning within prompt) of the rating on pattern presence and
log length, over completions with numeric ratings. Standard errors and the
honesty-minus-other difference are bootstrapped by prompt.

Usage:
    python3 aspect_ratings.py uf_raw/*.jsonl [--patterns rather_than not_only_just_merely_but ...] [--boot 500]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import compile_patterns

ASPECTS = ["honesty", "helpfulness", "instruction_following", "truthfulness"]
WORD_RE = re.compile(r"\w+")


def load(files, compiled, pats):
    prompts = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            rows = []
            for c in r.get("completions", []):
                ann = c.get("annotations", {})
                try:
                    ratings = [float(ann[a]["Rating"]) for a in ASPECTS]
                except (KeyError, ValueError, TypeError):
                    continue
                text = c.get("response") or ""
                if not text:
                    continue
                rows.append((ratings, [bool(compiled[p].search(text)) for p in pats],
                             np.log1p(len(WORD_RE.findall(text)))))
            if len(rows) >= 2:
                prompts.append(rows)
    return prompts


def fit(prompts, k):
    """Within-prompt OLS of each aspect rating on pattern k presence and log length; returns coef per aspect."""
    Y, X = [], []
    for rows in prompts:
        r = np.array([x[0] for x in rows])
        f = np.array([[float(x[1][k]), x[2]] for x in rows])
        Y.append(r - r.mean(axis=0))
        X.append(f - f.mean(axis=0))
    Y, X = np.vstack(Y), np.vstack(X)
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    return coef[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--patterns", nargs="+", default=["rather_than", "not_only_just_merely_but", "instead_of", "not_but"])
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    compiled = compile_patterns()
    prompts = load(args.files, compiled, args.patterns)
    n = sum(len(p) for p in prompts)
    print(f"{len(prompts)} prompts, {n} rated completions")
    rng = np.random.default_rng(args.seed)
    for k, pid in enumerate(args.patterns):
        users = sum(any(x[1][k] for x in p) and not all(x[1][k] for x in p) for p in prompts)
        est = fit(prompts, k)
        boots = []
        for _ in range(args.boot):
            idx = rng.integers(0, len(prompts), len(prompts))
            boots.append(fit([prompts[i] for i in idx], k))
        boots = np.array(boots)
        print(f"\n{pid}: {sum(x[1][k] for p in prompts for x in p)} completions use it; "
              f"{users} prompts with and without it")
        for a, name in enumerate(ASPECTS):
            lo, hi = np.percentile(boots[:, a], [2.5, 97.5])
            print(f"  {name:22s} {est[a]:+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")
        for a in range(1, len(ASPECTS)):
            d = boots[:, 0] - boots[:, a]
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  honesty - {ASPECTS[a]:22s} {est[0] - est[a]:+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")


if __name__ == "__main__":
    main()

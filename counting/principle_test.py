#!/usr/bin/env python3
r"""
Does instructing a model to be honest make it use "rather than" more? In the
raw UltraFeedback release, each completion was generated with a principle
(helpfulness, honesty, truthfulness, verbalized calibration, ...) sampled at
random and added to the system prompt; the record keeps the principle and
the exact system prompt. Specified before running: the prediction is that
completions under the honesty principle use "rather than" (and the other
corrective forms) more often than completions under helpfulness.

Because each source dataset draws from its own set of principles, the
comparison is made within source: share of completions using each pattern,
by principle, and a logistic regression of pattern presence on principle
(helpfulness as reference) with source and generating model as fixed
effects and log length as covariate.

Usage:
    python3 principle_test.py uf_raw/*.jsonl [--patterns rather_than not_only_just_merely_but ...]
"""
import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import compile_patterns

WORD_RE = re.compile(r"\w+")


def logistic(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1])
        b += np.linalg.solve(H, X.T @ (y - p))
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--patterns", nargs="+", default=["rather_than", "not_only_just_merely_but", "instead_of", "not_but"])
    args = ap.parse_args()
    compiled = compile_patterns()
    rows = []
    prompts = collections.defaultdict(set)
    for f in args.files:
        for line in open(f):
            r = json.loads(line)
            for c in r.get("completions", []):
                text = c.get("response") or ""
                if not text or not c.get("principle"):
                    continue
                rows.append(dict(source=r.get("source", Path(f).stem), principle=c["principle"], model=c.get("model", "?"),
                                 loglen=math.log1p(len(WORD_RE.findall(text))),
                                 **{p: bool(compiled[p].search(text)) for p in args.patterns}))
                prompts[(r.get("source", Path(f).stem), c["principle"])].add(c.get("custom_system_prompt", ""))
    print(f"{len(rows)} completions")
    by = collections.Counter((r["source"], r["principle"]) for r in rows)
    for (src, pr), n in sorted(by.items()):
        share = " ".join(f"{p}={100 * np.mean([r[p] for r in rows if r['source'] == src and r['principle'] == pr]):.2f}%"
                         for p in args.patterns)
        print(f"  {src:14s} {pr:24s} n={n:6d} system prompts={len(prompts[(src, pr)]):3d}  {share}")
    principles = sorted({r["principle"] for r in rows} - {"helpfulness"})
    sources = sorted({r["source"] for r in rows})
    models = sorted({r["model"] for r in rows})
    X = np.column_stack([np.ones(len(rows))]
                        + [[r["principle"] == p for r in rows] for p in principles]
                        + [[r["source"] == s for r in rows] for s in sources[1:]]
                        + [[r["model"] == m for r in rows] for m in models[1:]]
                        + [[r["loglen"] for r in rows]]).astype(float)
    for pat in args.patterns:
        y = np.array([r[pat] for r in rows], float)
        b, se = logistic(X, y)
        print(f"\n{pat}: odds ratio vs helpfulness principle (source, model fixed effects; log length)")
        for k, p in enumerate(principles, start=1):
            print(f"  {p:24s} {math.exp(b[k]):.2f} (95% CI {math.exp(b[k] - 1.96 * se[k]):.2f}-"
                  f"{math.exp(b[k] + 1.96 * se[k]):.2f}), p = {math.erfc(abs(b[k] / se[k]) / math.sqrt(2)):.3g}")


if __name__ == "__main__":
    main()

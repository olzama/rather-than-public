#!/usr/bin/env python3
r"""
Do preference datasets reward corrective framing? For each (chosen,
rejected) pair of responses to the same prompt, counts each antithesis
pattern (antithesis_patterns.py) in the final assistant turn of both.

Per dataset (and per source where the data has one):
  - pairs where only the chosen response uses the pattern vs. only the
    rejected one (discordant pairs; two-sided sign test);
  - a conditional logistic model of which response is chosen, on the
    difference in pattern presence and in log length (length is a known
    correlate of preference), fit on both orderings of each pair without
    an intercept; odds ratio for the pattern.

Reads Parquet files with pandas (needs pyarrow). Chosen/rejected columns are
either strings or lists of chat messages.

Usage:
    python3 preference_pairs.py --data ultrafeedback=uf_train_prefs.parquet \
        --data tulu3=tulu3_0.parquet,tulu3_1.parquet,... [--patterns rather_than instead_of ...]
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import compile_patterns

WORD_RE = re.compile(r"\w+")


def last_assistant(x):
    if isinstance(x, str):
        return x
    msgs = list(x)
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def conditional_logit(d_feat, d_len):
    """Fit P(chosen) on differences, both orderings, no intercept; returns coef and SE for the feature."""
    X = np.column_stack([np.concatenate([d_feat, -d_feat]), np.concatenate([d_len, -d_len])])
    y = np.concatenate([np.ones(len(d_feat)), np.zeros(len(d_feat))])
    b = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        b += np.linalg.solve(H, X.T @ (y - p))
    # each pair appears twice: SE from the information of one ordering
    se = np.sqrt(np.diag(np.linalg.inv(H / 2)))
    return b, se


def analyse(name, df, pats, compiled):
    ch = df["chosen"].map(last_assistant)
    rj = df["rejected"].map(last_assistant)
    keep = (ch.str.len() > 0) & (rj.str.len() > 0) & (ch != rj)
    ch, rj = ch[keep], rj[keep]
    lc = np.log1p(ch.map(lambda t: len(WORD_RE.findall(t))).to_numpy())
    lr = np.log1p(rj.map(lambda t: len(WORD_RE.findall(t))).to_numpy())
    print(f"\n== {name}: {len(ch)} pairs; median words chosen {np.expm1(np.median(lc)):.0f}, rejected {np.expm1(np.median(lr)):.0f}")
    print(f"  {'pattern':28s} {'chosen':>8s} {'rejected':>8s} {'only ch':>8s} {'only rj':>8s} {'sign p':>8s} {'OR|len':>7s} {'95% CI':>13s}")
    for pid in pats:
        rx = compiled[pid]
        c = ch.map(lambda t: bool(rx.search(t))).to_numpy()
        r = rj.map(lambda t: bool(rx.search(t))).to_numpy()
        oc, orj = int((c & ~r).sum()), int((r & ~c).sum())
        p = stats.binomtest(oc, oc + orj).pvalue if oc + orj else float("nan")
        b, se = conditional_logit(c.astype(float) - r.astype(float), lc - lr)
        print(f"  {pid:28s} {c.mean():8.2%} {r.mean():8.2%} {oc:8d} {orj:8d} {p:8.2g} {math.exp(b[0]):7.2f} "
              f"{math.exp(b[0] - 1.96 * se[0]):6.2f}-{math.exp(b[0] + 1.96 * se[0]):5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="NAME=file1.parquet[,file2...]")
    ap.add_argument("--patterns", nargs="+",
                    default=["rather_than", "instead_of", "x_not_y", "not_but", "not_only_just_merely_but"])
    ap.add_argument("--by-source", action="store_true")
    args = ap.parse_args()
    compiled = compile_patterns()
    for spec in args.data:
        name, files = spec.split("=", 1)
        df = pd.concat([pd.read_parquet(f, columns=None) for f in files.split(",")], ignore_index=True)
        analyse(name, df, args.patterns, compiled)
        if args.by_source and "source" in df:
            for src, sub in df.groupby("source"):
                if len(sub) >= 2000:
                    analyse(f"{name} / {src}", sub, args.patterns, compiled)


if __name__ == "__main__":
    main()

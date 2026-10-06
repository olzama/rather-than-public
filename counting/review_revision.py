#!/usr/bin/env python3
r"""
Does the review drive the construction into the revision? For each paper in
dataset (3) and each model: the review (evaluated_papers/), the generated
paper and the revision. Review measures: words in the Weaknesses and
Suggestions sections, and the number of claim/evidence criticisms (CUES,
fixed before running). Outcome: "rather than" occurrences added by the
revision (revised - generated), and the change in rate per 1,000 words.

Reports Spearman correlations of each review measure with the added count,
and a regression of the added count on the review measure with the change
in log length and the generated paper's count as covariates. The review
prompt contains "rather than" for every paper alike, so it cannot produce a
per-paper correlation.

Usage:
    python3 review_revision.py ../data/generated/gpt_papers/batch2 ../data/generated/gpt_papers/batch3 [--excluded ../data/excluded_documents.tsv] \
        [--models gpt-5.6-sol gpt-6-sol]
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

RT = re.compile(r"\brather than\b", re.I)
WORD = re.compile(r"\w+")
CUES = re.compile(r"overstat|overclaim|over-claim|unsupported|not (?:well[- ])?supported|unsubstantiated|speculative"
                  r"|lack(?:s|ing)? (?:of )?(?:\w+ )?(?:evidence|results|validation|experiments|support)"
                  r"|insufficient (?:\w+ )?(?:evidence|support|detail)|no (?:quantitative|empirical|experimental) "
                  r"|does not (?:report|provide|establish|demonstrate|support|justify|show)"
                  r"|(?:claims?|conclusions?) (?:are|is|seem|appear) (?:not|too|overly)", re.I)


def section(text, names):
    """Text under the given plain-text headings, up to the next known heading."""
    heads = ["Summary", "Strengths", "Weaknesses", "Suggestions for Improvement", "Questions for the Authors",
             "Overall Assessment"]
    out, keep = [], False
    for line in text.splitlines():
        h = line.strip().strip("#*: ").strip()
        if h in heads:
            keep = h in names
            continue
        if keep:
            out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generation_dirs", nargs="+", help="one or more generation runs (data/generated/gpt_papers/batch*)")
    ap.add_argument("--excluded")
    ap.add_argument("--models", nargs="+", default=["gpt-5.6-sol", "gpt-6-sol"])
    args = ap.parse_args()
    from generation_view import combined
    g = combined(args.generation_dirs, args.models)
    skip = set()
    if args.excluded:
        skip = {l.split("\t")[1] for l in list(open(args.excluded))[1:] if l.split("\t")[0] == "acl2019"}
    for m in args.models:
        ids = sorted({p.stem for p in (g / m / "generated_papers").glob("*.md")}
                     & {p.stem for p in (g / m / "revised_papers").glob("*.md")}
                     & {p.stem for p in (g / m / "evaluated_papers").glob("*.md")} - skip)
        rows = []
        for i in ids:
            gen = (g / m / "generated_papers" / f"{i}.md").read_text(errors="replace")
            rev = (g / m / "revised_papers" / f"{i}.md").read_text(errors="replace")
            review = (g / m / "evaluated_papers" / f"{i}.md").read_text(errors="replace")
            crit = section(review, {"Weaknesses", "Suggestions for Improvement"})
            wg, wr = len(WORD.findall(gen)), len(WORD.findall(rev))
            rows.append(dict(crit_words=len(WORD.findall(crit)), cues=len(CUES.findall(crit)),
                             added=len(RT.findall(rev)) - len(RT.findall(gen)), gen_rt=len(RT.findall(gen)),
                             d_rate=1000 * len(RT.findall(rev)) / max(wr, 1) - 1000 * len(RT.findall(gen)) / max(wg, 1),
                             d_len=math.log(max(wr, 1)) - math.log(max(wg, 1))))
        print(f"\n== {m}: {len(rows)} papers; criticism words median {np.median([r['crit_words'] for r in rows]):.0f}, "
              f"claim/evidence cues median {np.median([r['cues'] for r in rows]):.0f} (papers with none: "
              f"{sum(r['cues'] == 0 for r in rows)}); added 'rather than' mean {np.mean([r['added'] for r in rows]):+.1f}")
        for meas in ("crit_words", "cues"):
            x = [r[meas] for r in rows]
            for out in ("added", "d_rate"):
                rho, p = stats.spearmanr(x, [r[out] for r in rows])
                print(f"  Spearman {meas:10s} vs {out:6s}: rho {rho:+.2f}, p = {p:.3g}")
            X = np.column_stack([np.ones(len(rows)), x, [r["d_len"] for r in rows], [r["gen_rt"] for r in rows]])
            y = np.array([r["added"] for r in rows], float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ coef
            se = np.sqrt(np.diag(res @ res / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
            print(f"  OLS added ~ {meas} + d_log_length + generated count: {meas} coef {coef[1]:+.3f} "
                  f"(SE {se[1]:.3f}), p = {math.erfc(abs(coef[1] / se[1]) / math.sqrt(2)):.3g}")


if __name__ == "__main__":
    main()

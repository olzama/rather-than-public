#!/usr/bin/env python3
r"""
When an LLM's revision removes "rather than", does another antithesis form
take its place? For each model, compares each paper's generated and revised
versions (paper_pipeline output, Markdown, made plain as in
llm_instances.py) in two ways:

  paper level     per-paper change (revised - generated) in the rate per
                  1,000 words of every antithesis pattern
                  (antithesis_patterns.py); mean change, Wilcoxon signed-rank
                  test, and Spearman correlation of each pattern's change
                  with the change in "rather than";
  sentence level  every generated sentence with "rather than" is matched to
                  its closest revised sentence (difflib ratio >= --min-sim);
                  for sentences whose match lacks "rather than" (the phrase
                  was removed while the sentence survived), the antithesis
                  patterns in the revised sentence are counted. The same is
                  done in reverse for "rather than" added in revision.

Usage:
    python3 revision_substitution.py ../data/generated/gpt_papers/batch2 ../data/generated/gpt_papers/batch3 \
        [--models gpt-4o gpt-5.6-sol gpt-6-sol] [--min-sim 0.6]
"""
import argparse
import collections
import difflib
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import PATTERNS  # noqa: E402
from llm_instances import plain, redact  # noqa: E402

RX = {pid: re.compile(p, re.I) for pid, _tier, _label, p, _note in PATTERNS}
LABEL = {pid: label for pid, _tier, label, _p, _note in PATTERNS}
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
WORD = re.compile(r"\w+")


def load(path):
    return " ".join(redact(plain(path.read_text(errors="ignore"))).split())


def rates(text):
    n = len(WORD.findall(text)) or 1
    return {pid: 1000 * len(rx.findall(text)) / n for pid, rx in RX.items()}


def patterns_in(sentence):
    return [pid for pid, rx in RX.items() if rx.search(sentence)]


def match(sentence, candidates, min_sim):
    best, best_r = None, min_sim
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(sentence)
    for c in candidates:
        sm.set_seq1(c)
        if sm.real_quick_ratio() < best_r or sm.quick_ratio() < best_r:
            continue
        r = sm.ratio()
        if r >= best_r:
            best, best_r = c, r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generation_dirs", nargs="+")
    ap.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-5.6-sol", "gpt-6-sol"])
    ap.add_argument("--min-sim", type=float, default=0.6)
    args = ap.parse_args()
    for m in args.models:
        pairs = []
        for gdir in map(Path, args.generation_dirs):
            for g in sorted((gdir / m / "generated_papers").glob("*.md")):
                r = gdir / m / "revised_papers" / g.name
                if r.exists():
                    pairs.append((load(g), load(r)))
        if not pairs:
            continue
        print(f"\n== {m}: {len(pairs)} papers (generated -> revised)")
        dg = [rates(g) for g, _ in pairs]
        dr = [rates(r) for _, r in pairs]
        delta = {pid: np.array([b[pid] - a[pid] for a, b in zip(dg, dr)]) for pid in RX}
        print(f"  {'pattern':<30}{'generated':>10}{'revised':>9}{'change':>8}{'p':>9}{'rho w/ RT':>11}")
        for pid in RX:
            a = np.mean([x[pid] for x in dg])
            b = np.mean([x[pid] for x in dr])
            nz = delta[pid][delta[pid] != 0]
            p = stats.wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
            rho = stats.spearmanr(delta["rather_than"], delta[pid])[0] if pid != "rather_than" else float("nan")
            print(f"  {LABEL[pid]:<30}{a:>10.3f}{b:>9.3f}{b - a:>+8.3f}{p:>9.3g}{rho:>+11.2f}")

        for direction, (src_i, dst_i) in (("removed", (0, 1)), ("added", (1, 0))):
            found = collections.Counter()
            n_src = n_kept = n_unmatched = 0
            for pair in pairs:
                src = SENT.split(pair[src_i])
                dst = SENT.split(pair[dst_i])
                for s in src:
                    if not RX["rather_than"].search(s):
                        continue
                    n_src += 1
                    t = match(s, dst, args.min_sim)
                    if t is None:
                        n_unmatched += 1
                    elif RX["rather_than"].search(t):
                        n_kept += 1
                    else:
                        found.update(patterns_in(t) or ["none"])
            n_changed = n_src - n_kept - n_unmatched
            what = "in the revised sentence" if direction == "removed" else "in the generated sentence it replaced"
            side = "generated" if direction == "removed" else "revised"
            print(f"  sentences with 'rather than' in the {side} version: {n_src}; matched and kept {n_kept}, "
                  f"matched but phrase {direction} {n_changed}, no close match {n_unmatched}")
            if n_changed:
                parts = ", ".join(f"{(LABEL.get(k, k))} {v}" for k, v in found.most_common() if v > 0)
                print(f"    antithesis forms {what} where 'rather than' was {direction}: {parts}")


if __name__ == "__main__":
    main()

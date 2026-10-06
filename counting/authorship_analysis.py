#!/usr/bin/env python3
r"""
Authorship-guess task: do annotators who believe a paper's passage was
written by an LLM also find that paper's "rather than" uses annoying?

Inputs: the passages (authorship_passages.py provenance: passage -> pool
item -> paper), each annotator's guesses (sheet rows "auth:<pid>", pulled
with pull_sheet_labels.py into --guesses, or read from a sheet CSV export
with --csv), and the annotators' annoyance labels (human_labels.jsonl).

Belief score per guess: human 0, probably human 1, probably LLM 2, LLM 3.
Per annotator:
  - item level: for each passage, the annotator's label on the pool item it
    was drawn next to (annoying vs. not); annoying rate by belief, and a
    logistic regression of annoying on belief (and on passage kind: seen
    vs. vicinity);
  - paper level: mean belief per paper vs. share of that paper's pool items
    the annotator labeled annoying (Spearman).
Fillers (kind "filler", pool_authorship_passages.py) contain no antithesis
construction; they are left out of the item and paper analyses and compared
with the item passages from the same source, to test whether the
construction itself raises the "LLM" hunch.
Predictions for the pool task (../data/authorship_pool; reduced on 2026-10-01 to
303 passages: all annoying items, 50 other arXiv 2026, 30 ACL 2019 and 30 GPT
items, and 101 fillers), specified before any guesses were collected
(2026-10-01), all one-sided:
  P1  passages of items A1 or A2 labeled annoying get higher LLM beliefs than
      passages of items they labeled legitimate (each label set: A1,
      A2, Either);
  P2  passages with the construction get higher LLM beliefs than fillers
      from the same source;
  P3  calibration: GPT items get higher LLM beliefs than ACL 2019 items,
      overall and for GPT items written from ACL 2019 papers only (same
      topics and era, so "sounds like 2019" cannot pass for "sounds human").
Calibration: when the passages include items of known authorship (GPT items,
corpus llm_*; ACL 2019 items), each guesser's beliefs on the two are compared
(mean, AUC, one-sided Mann-Whitney).
Test names (zz-test...) are ignored.

With --labels, each guesser's guesses are related to other annotators'
annoyance labels: e.g. guessers who never judged annoyance against A1's and
A2's labels. "Either" is the union label set of A1 and A2 (annoying if
either labeled the item annoying, legitimate if both labeled it legitimate).
Guesses come from a sheet CSV export (--csv) or straight from the sheet
backend (--sheet-url).

Usage:
    python3 authorship_analysis.py ../data/authorship ../data/annotation --csv labels.csv \
        [--annotators A1 A2]
    python3 authorship_analysis.py ../data/authorship ../data/annotation --sheet-url URL \
        --annotators A3 A5 --labels A1 A2 Either
"""
import argparse
import collections
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

BELIEF = {"human": 0, "probably_human": 1, "probably_llm": 2, "llm": 3}
VERSION_RE = re.compile(r"_v\d+$")


def logistic(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
        b += np.linalg.solve(H, X.T @ (y - p))
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def load_guesses(path_csv):
    """annotator -> pid -> label, latest row wins (sheet CSV export)."""
    g = collections.defaultdict(dict)
    for r in csv.DictReader(open(path_csv)):
        name, item = r["annotator"].strip(), r["item_id"]
        if item.startswith("auth:") and not name.lower().startswith("zz-test"):
            g[name][item[5:]] = r["label"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("authorship_dir")
    ap.add_argument("annotation_dir")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="sheet export (received_at, annotator, item_id, label, ...)")
    src.add_argument("--sheet-url", help="sheet backend URL (sheet_backend.gs)")
    ap.add_argument("--token", default="rt-2026-annotate")
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"], help="guessers")
    ap.add_argument("--labels", nargs="+", help="label sets to relate guesses to (default: each guesser's own)")
    args = ap.parse_args()
    prov = {r["pid"]: r for r in map(json.loads, open(Path(args.authorship_dir) / "passages_provenance.jsonl"))}
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(Path(args.annotation_dir) / "human_labels.jsonl")):
        labels[r["annotator"]][r["item_id"]] = r["label"]
    pool_prov = {r["item_id"]: r for r in map(json.loads, open(Path(args.annotation_dir) / "items_provenance.jsonl"))}
    if "A1" in labels and "A2" in labels:
        for i in set(labels["A1"]) & set(labels["A2"]):
            pair = (labels["A1"][i], labels["A2"][i])
            if "annoying" in pair:
                labels["Either"][i] = "annoying"
            elif pair == ("legitimate", "legitimate"):
                labels["Either"][i] = "legitimate"
    if args.csv:
        guesses = load_guesses(args.csv)
    else:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
        from pull_sheet_labels import fetch
        guesses = {n: {k[5:]: v for k, (v, _ts) in fetch(args.sheet_url, args.token, n).items() if k.startswith("auth:")}
                   for n in args.annotators}
    paper = lambda item: (pool_prov[item]["corpus"] == "acl2019", VERSION_RE.sub("", pool_prov[item]["doc_id"]))
    for guesser in args.annotators:
        g = guesses.get(guesser, {})
        known = [(p["corpus"].startswith("llm_"), BELIEF[g[pid]]) for pid, p in prov.items()
                 if g.get(pid) in BELIEF and (p["corpus"].startswith("llm_") or p["corpus"] == "acl2019")]
        llm = [b for is_llm, b in known if is_llm]
        hum = [b for is_llm, b in known if not is_llm]
        if llm and hum:
            u = stats.mannwhitneyu(llm, hum, alternative="greater")
            print(f"\n== calibration, {guesser}: mean belief GPT {np.mean(llm):.2f} (n={len(llm)}) vs ACL 2019 "
                  f"{np.mean(hum):.2f} (n={len(hum)}); AUC {u.statistic / (len(llm) * len(hum)):.2f}, p = {u.pvalue:.3g}")
            same_era = [BELIEF[g[pid]] for pid, p in prov.items() if g.get(pid) in BELIEF
                        and p["corpus"].startswith("llm_") and not re.search(r"__\d{4}\.\d{4,5}$", p["doc_id"])]
            if same_era:
                u = stats.mannwhitneyu(same_era, hum, alternative="greater")
                print(f"   GPT written from ACL 2019 papers {np.mean(same_era):.2f} (n={len(same_era)}) vs ACL 2019: "
                      f"AUC {u.statistic / (len(same_era) * len(hum)):.2f}, p = {u.pvalue:.3g}")
        group = lambda c: "GPT" if c.startswith("llm_") else ("ACL 2019" if c == "acl2019" else "arXiv 2026")
        for grp in ("ACL 2019", "arXiv 2026", "GPT"):
            it = [BELIEF[g[pid]] for pid, p in prov.items() if g.get(pid) in BELIEF and group(p["corpus"]) == grp
                  and p["kind"] != "filler"]
            fi = [BELIEF[g[pid]] for pid, p in prov.items() if g.get(pid) in BELIEF and group(p["corpus"]) == grp
                  and p["kind"] == "filler"]
            if it and fi:
                u = stats.mannwhitneyu(it, fi, alternative="two-sided")
                print(f"   {grp}: mean belief, passages with the construction {np.mean(it):.2f} (n={len(it)}) "
                      f"vs fillers {np.mean(fi):.2f} (n={len(fi)}), p = {u.pvalue:.3g}")
    print("\n== committed predictions (one-sided Mann-Whitney)")
    for guesser in args.annotators:
        g = guesses.get(guesser, {})
        for name in (args.labels or []):
            ann = [BELIEF[g[pid]] for pid, p in prov.items() if g.get(pid) in BELIEF and p["kind"] != "filler"
                   and labels[name].get(p["item_id"]) == "annoying"]
            leg = [BELIEF[g[pid]] for pid, p in prov.items() if g.get(pid) in BELIEF and p["kind"] != "filler"
                   and labels[name].get(p["item_id"]) == "legitimate" and not p["corpus"].startswith("llm_")
                   and p["corpus"] != "acl2019"]
            if ann and leg:
                u = stats.mannwhitneyu(ann, leg, alternative="greater")
                print(f"  P1 {guesser}, {name} labels: annoying {np.mean(ann):.2f} (n={len(ann)}) vs legitimate arXiv 2026 "
                      f"{np.mean(leg):.2f} (n={len(leg)}); AUC {u.statistic / (len(ann) * len(leg)):.2f}, p = {u.pvalue:.3g}")
        src = collections.defaultdict(lambda: ([], []))
        for pid, p in prov.items():
            if g.get(pid) in BELIEF:
                src[p["corpus"]][p["kind"] == "filler"].append(BELIEF[g[pid]])
        it = [b for c, (a, f) in src.items() if f for b in a]
        fi = [b for c, (a, f) in src.items() if a for b in f]
        if it and fi:
            u = stats.mannwhitneyu(it, fi, alternative="greater")
            print(f"  P2 {guesser}: construction {np.mean(it):.2f} (n={len(it)}) vs fillers {np.mean(fi):.2f} (n={len(fi)}), "
                  f"sources with both; AUC {u.statistic / (len(it) * len(fi)):.2f}, p = {u.pvalue:.3g}")
    for guesser, name in [(gs, ls) for gs in args.annotators for ls in (args.labels or [gs])]:
        g = guesses.get(guesser, {})
        rows = [(BELIEF[g[pid]], labels[name].get(p["item_id"]), p["kind"], p["corpus"])
                for pid, p in prov.items() if g.get(pid) in BELIEF and p["kind"] != "filler"]
        rows = [r for r in rows if r[1] in ("annoying", "legitimate", "unsure")]
        print(f"\n== guesses: {guesser} ({len(g)}); labels: {name}; {len(rows)} passages with a label on their pool item")
        if len(rows) < 10:
            continue
        dist = collections.Counter(r[0] for r in rows)
        print("  belief distribution (0=human .. 3=LLM):", dict(sorted(dist.items())))
        for bval in range(4):
            sub = [r for r in rows if r[0] == bval]
            if sub:
                k = sum(r[1] == "annoying" for r in sub)
                print(f"  belief {bval}: item annoying {k}/{len(sub)} ({k / len(sub):.0%})")
        y = np.array([r[1] == "annoying" for r in rows], float)
        X = np.column_stack([np.ones(len(rows)), [r[0] for r in rows], [r[2] == "seen" for r in rows],
                             [r[3] == "acl2019" for r in rows]]).astype(float)
        b, se = logistic(X, y)
        ex = lambda v: math.exp(min(v, 50))
        print(f"  logistic: odds ratio per belief step {ex(b[1]):.2f} "
              f"(95% CI {ex(b[1] - 1.96 * se[1]):.2f}-{ex(b[1] + 1.96 * se[1]):.2f}), "
              f"p = {math.erfc(abs(b[1] / se[1]) / math.sqrt(2)):.3g}  (controls: seen passage, ACL 2019)")
        for kind in ("vicinity", "seen"):
            sub = [r for r in rows if r[2] == kind]
            if len(sub) > 5:
                rho, p = stats.spearmanr([r[0] for r in sub], [r[1] == "annoying" for r in sub])
                print(f"  {kind:8s} passages: n={len(sub)}, Spearman belief vs annoying {rho:+.2f}, p = {p:.3g}")
        per = collections.defaultdict(list)
        for pid, p in prov.items():
            if g.get(pid) in BELIEF and p["item_id"] in pool_prov and p["kind"] != "filler":
                per[paper(p["item_id"])].append(BELIEF[g[pid]])
        pairs = []
        for pk, bel in per.items():
            items = [i for i in labels[name] if i in pool_prov and "repeat_of" not in pool_prov[i]
                     and paper(i) == pk and labels[name][i] != "garbled"]
            if items:
                pairs.append((np.mean(bel), np.mean([labels[name][i] == "annoying" for i in items])))
        if len(pairs) > 5:
            rho, p = stats.spearmanr(*zip(*pairs))
            print(f"  paper level: {len(pairs)} papers, Spearman mean belief vs share annoying {rho:+.2f}, p = {p:.3g}")


if __name__ == "__main__":
    main()

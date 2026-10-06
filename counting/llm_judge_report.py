#!/usr/bin/env python3
r"""
Compare LLM judgments (llm_judge.py output) with one human annotator's
labels. For each judgment file: agreement and Cohen's kappa on
annoying-vs-not, precision/recall of the model's "annoying" label, the
AUC of its 1-5 score for the human's annoying items, and the model's
annoying rate for ACL 2019 vs. pooled arXiv 2026.

The deletion prompt's "redundant" label counts as annoying (its score
already runs from 1 = needed to 5 = redundant). Items the human labeled
garbled are left out. Every file is also scored on the items common to
all files, so prompts evaluated on different item sets (few-shot runs
hold out their example items) can be compared directly.

Usage:
    python3 llm_judge_report.py ../data/annotation <judgments_dir> [--annotator A1]
"""
import argparse
import collections
import json
from pathlib import Path

from scipy import stats

POSITIVE = {"annoying", "redundant"}
ARXIV2026 = {"arxiv2026", "arxiv_v1", "arxiv_latest"}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def auc(scores, y):
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def evaluate(rows, human, prov):
    h = [human[r["item_id"]] == "annoying" for r in rows]
    j = [r["label"] in POSITIVE for r in rows]
    tp = sum(a and b for a, b in zip(h, j))
    rate = collections.defaultdict(lambda: [0, 0])
    for r, flag in zip(rows, j):
        c = prov[r["item_id"]]["corpus"]
        g = "acl2019" if c == "acl2019" else "arxiv2026" if c in ARXIV2026 else None
        if g:
            rate[g][0] += flag
            rate[g][1] += 1
    a, b = rate["acl2019"], rate["arxiv2026"]
    p = stats.fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1] if a[1] and b[1] else float("nan")
    return {
        "n": len(rows), "human_annoying": sum(h), "flagged": sum(j), "both": tp,
        "precision": tp / sum(j) if sum(j) else float("nan"), "recall": tp / sum(h) if sum(h) else float("nan"),
        "kappa": kappa(h, j), "auc": auc([r["score"] for r in rows], h),
        "acl2019": f"{a[0]}/{a[1]}", "arxiv2026": f"{b[0]}/{b[1]}", "fisher_p": p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("judgments_dir")
    ap.add_argument("--annotator", default="A1")
    args = ap.parse_args()
    d = Path(args.annotation_dir)

    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl")) if "excluded" not in r}
    human = {}
    for l in open(d / "human_labels.jsonl"):
        r = json.loads(l)
        if r["annotator"] == args.annotator and r["label"] != "garbled":
            human[r["item_id"]] = r["label"]

    runs = {}
    for f in sorted(Path(args.judgments_dir).glob("*.jsonl")):
        runs[f.stem] = [r for r in map(json.loads, open(f)) if r["item_id"] in human]
    common = set.intersection(*({r["item_id"] for r in rows} for rows in runs.values()))

    cols = ["n", "human_annoying", "flagged", "both", "precision", "recall", "kappa", "auc", "acl2019", "arxiv2026", "fisher_p"]
    for scope in ("all items judged", f"common items (n={len(common)})"):
        print(f"\n== {scope}")
        print("run".ljust(44) + "".join(c.rjust(max(10, len(c) + 1)) for c in cols))
        for name, rows in runs.items():
            if scope.startswith("common"):
                rows = [r for r in rows if r["item_id"] in common]
            m = evaluate(rows, human, prov)
            print(name.ljust(44) + "".join(
                (f"{m[c]:.2f}" if isinstance(m[c], float) else str(m[c])).rjust(max(10, len(c) + 1)) for c in cols))


if __name__ == "__main__":
    main()

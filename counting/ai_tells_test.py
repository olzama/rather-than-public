#!/usr/bin/env python3
r"""
Do other AI tells in the text an annotator saw go with annoyance?
Specified before running:

  Items     arXiv 2026 pool items (dataset (2), v1, latest, high-count),
            hidden repeats and excluded items left out, items the
            annotator labeled garbled left out.
  Text      what the annotator saw: the sentence and the sentences before
            and after it.
  Tells     (1) Kobak: the words annotated "style" in Kobak et al. (2025)'s
            excess-vocabulary list (../data/ai_tells/kobak_excess_words.csv),
            as a rate per 100 words.
            (2) Graphite: the named tell sets of Druck, Paredes and Smith
            (2026), GRAPHITE below, as the number of matches. Their
            "corrective framing" set is left out: it is the antithesis
            family, tested separately.
            (3) Initial emphatic adverb: a sentence opening with one of
            INITIAL_ADVERBS followed by a comma.
            Em dashes are counted but too rare in the extracted text to test.
  Primary   for each label set (each annotator; Either = annoying if either
            annotator labeled it so): logistic regression of annoying on the
            Kobak rate, the number of Graphite matches (log(1+n)) and the
            presence of an initial emphatic adverb, with log shown-text length
            and stratum fixed effects. The hypothesis is directional: more
            tells, more annoying (one-sided p per coefficient).
  Secondary Fisher's exact test, any Graphite tell vs. none; the annoying
            and legitimate means of each measure.

Usage:
    python3 ai_tells_test.py ../data/annotation --pair A1 A2 [--batch 1]
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

ARXIV = ["arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"]
KOBAK = Path(__file__).resolve().parent.parent / "data" / "ai_tells" / "kobak_excess_words.csv"

# Druck, Paredes and Smith (2026), quoted from the article's figure notes
# (marketing, hype, intensifiers, precision, contractions omitted) and its
# tables of tells (evaluative adjectives, transitions, avoiding tradeoffs,
# flagging importance).
GRAPHITE = {
    "marketing": ["payoff", "unlock", "elevate", "amplify", "streamline", "boost", "leverage", "wins", "tangible"],
    "hype": ["groundbreaking", "paramount", "unprecedented", "revolutionary", "invaluable", "profound", "exceptional"],
    "intensifiers": ["absolutely", "entirely", "highly", "incredibly"],
    "precision": ["exactly", "precisely", "specifically"],
    "evaluative": ["dependable", "practical", "meaningful", "deliberate", "measured", "steady", "immense"],
    "transitions": ["another dimension", "together these", "what comes next", "looking ahead the",
                    "furthermore the", "ultimately this", "additionally the"],
    "tradeoffs": ["without requiring", "without losing", "without sacrificing", "without compromising"],
    "importance": ["distinction matters", "matters because", "matters more than", "absolutely essential",
                   "remarkably", "single most", "arguably the most", "every single", "enormously"],
}
INITIAL_ADVERBS = ["crucially", "notably", "importantly", "ultimately", "critically", "fundamentally",
                   "additionally", "furthermore", "moreover", "remarkably", "interestingly"]
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
EM_DASH_RE = re.compile(r"—|---")


def phrase_re(phrases):
    return re.compile(r"\b(?:" + "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in phrases) + r")\b", re.I)


GRAPHITE_RE = phrase_re([p for ps in GRAPHITE.values() for p in ps])
INITIAL_RE = re.compile(r"(?:^|[.!?]\s+)(?:" + "|".join(INITIAL_ADVERBS) + r"),", re.I)


def measures(item, kobak):
    text = " ".join(x for x in (item.get("context_before", ""), item["sentence"], item.get("context_after", "")) if x)
    words = WORD_RE.findall(text.lower())
    n = max(len(words), 1)
    return {"kobak": 100 * sum(w in kobak for w in words) / n,
            "graphite": len(GRAPHITE_RE.findall(text)),
            "initial": bool(INITIAL_RE.search(text)),
            "em_dash": len(EM_DASH_RE.findall(text)),
            "length": n}


def logistic(X, y, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        b += np.linalg.solve(H, X.T @ (y - p))
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--pair", nargs=2, default=["A1", "A2"])
    ap.add_argument("--batch", type=int)
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    kobak = {r["word"] for r in csv.DictReader(open(KOBAK)) if r["type"] == "style"}
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl"))
            if r["corpus"] in ARXIV and "repeat_of" not in r and "excluded" not in r}
    pub = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl"))}
    items = {i for i in prov if i in pub and (args.batch is None or pub[i].get("batch", 1) == args.batch)}
    m = {i: measures(pub[i], kobak) for i in items}
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        if r["item_id"] in items:
            labels[r["annotator"]][r["item_id"]] = r["label"]
    a, b = args.pair
    labels["Either"] = {i: ("annoying" if "annoying" in (labels[a][i], labels[b][i]) else "other")
                        for i in labels[a] if i in labels[b] and "garbled" not in (labels[a][i], labels[b][i])}

    print(f"items: {len(items)} arXiv 2026{'' if args.batch is None else f' (batch {args.batch})'}; "
          f"{len(kobak)} Kobak style words")
    print(f"items with any Graphite tell: {sum(v['graphite'] > 0 for v in m.values())}; "
          f"initial emphatic adverb: {sum(v['initial'] for v in m.values())}; "
          f"em dash: {sum(v['em_dash'] > 0 for v in m.values())}")
    for name in (a, b, "Either"):
        rows = [i for i, lab in labels[name].items() if lab != "garbled"]
        y = np.array([labels[name][i] == "annoying" for i in rows], float)
        feats = [("Kobak rate", [m[i]["kobak"] for i in rows]),
                 ("Graphite log(1+n)", [math.log1p(m[i]["graphite"]) for i in rows]),
                 ("initial adverb", [float(m[i]["initial"]) for i in rows])]
        X = np.column_stack([np.ones(len(rows))] + [f for _, f in feats]
                            + [[math.log(m[i]["length"]) for i in rows]]
                            + [[float(prov[i]["corpus"] == s) for i in rows] for s in ARXIV[1:]])
        keep = [True] * 5 + [X[:, k].sum() > 0 for k in range(5, X.shape[1])]
        coef, se = logistic(X[:, keep], y)
        print(f"\n{name} (n={len(rows)}, annoying {int(y.sum())})")
        for k, (fname, f) in enumerate(feats, start=1):
            f = np.array(f)
            z = coef[k] / se[k]
            print(f"  {fname:18s} mean annoying {f[y == 1].mean():.3f} vs other {f[y == 0].mean():.3f}; "
                  f"coef {coef[k]:+.3f} (SE {se[k]:.3f}), one-sided p = {1 - stats.norm.cdf(z):.3g}")
        g = np.array([m[i]["graphite"] > 0 for i in rows])
        k1, n1, k0, n0 = int(y[g].sum()), int(g.sum()), int(y[~g].sum()), int((~g).sum())
        print(f"  any Graphite tell: {k1}/{n1} ({k1 / max(n1, 1):.0%}) vs none {k0}/{n0} ({k0 / n0:.0%}); "
              f"Fisher p = {stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])[1]:.3g}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Compare the vocabulary of annoying vs. legitimate "rather than" items across
all words and word sequences, with no predefined word lists.

Items are the human-labeled arXiv 2026 pool items (dataset (2), v1, latest,
and high-count strata; ACL 2019 has no annoying items, so including it would
contrast years, not labels). Hidden repeats, unsure and garbled items are
left out. Terms are lowercased word tokens and punctuation, as 1-3-grams,
counted separately in the "rather than" sentence and in the surrounding
context (the sentence before and after).

Each term is scored by the log-odds ratio of the share of annoying vs.
legitimate items that contain it (presence, not counts, so sentence length
and repeated terms within one item do not drive the score), smoothed toward
the term's share of all "rather than" instances in dataset (2), after the
informative-prior method of Monroe, Colaresi & Quinn (2008). Positive z = more
typical of annoying items. Benjamini-Hochberg q-values are computed over all
terms of both windows (two-sided normal p from z). Terms occurring in fewer than --min-items items
are skipped. For each top term the report gives how many annoying and
legitimate items contain it, and example snippets from annoying items.

Usage:
    python3 lexical_contrast.py ../data/annotation ../data/antithesis_instances/arxiv2026.jsonl \
        [--annotator A1] [--top 30] [--min-items 4] [--out report.md]
"""
import argparse
import collections
import json
import math
import re
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[^\sa-z0-9]")
ARXIV2026 = {"arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"}


def tokens(text):
    return TOKEN_RE.findall(text.lower())


def ngrams(toks, n_max=3):
    out = []
    for n in range(1, n_max + 1):
        out += [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return out


def windows(rec):
    return {
        "sentence": rec["sentence"],
        "context": (rec.get("context_before", "") + " \n " + rec.get("context_after", "")),
    }


def log_odds(f1, n1, f2, n2, prior_p, strength):
    """Per-term z-scores for the difference in the share of items containing
    the term (log-odds), smoothed toward the term's share of prior items with
    `strength` pseudo-items (after Monroe, Colaresi & Quinn 2008)."""
    z = {}
    for w, p in prior_p.items():
        a, b = strength * p, strength * (1 - p)
        y1, y2 = f1.get(w, 0), f2.get(w, 0)
        d = math.log((y1 + a) / (n1 - y1 + b)) - math.log((y2 + a) / (n2 - y2 + b))
        var = 1 / (y1 + a) + 1 / (n1 - y1 + b) + 1 / (y2 + a) + 1 / (n2 - y2 + b)
        z[w] = d / math.sqrt(var)
    return z


def snippet(text, term, width=70):
    toks_text = text.replace("\n", " ")
    m = re.search(r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"\s*") + r"(?![a-z0-9])", toks_text.lower())
    if not m:
        return toks_text[:2 * width]
    a, b = max(0, m.start() - width), min(len(toks_text), m.end() + width)
    return ("…" if a else "") + toks_text[a:m.start()] + "**" + toks_text[m.start():m.end()] + "**" + toks_text[m.end():b] + ("…" if b < len(toks_text) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--corpora", nargs="+", default=sorted(ARXIV2026), help="corpora whose items are compared")
    ap.add_argument("prior_instances")
    ap.add_argument("--annotator", default="A1")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-items", type=int, default=4)
    ap.add_argument("--prior-strength", type=float, default=10,
                    help="weight of the prior, in pseudo-items per group")
    ap.add_argument("--out", help="write a markdown report here (default: print)")
    args = ap.parse_args()
    d = Path(args.annotation_dir)

    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl")) if "excluded" not in r}
    pub = {json.loads(l)["item_id"]: json.loads(l) for l in open(d / "items_public.jsonl")}
    pub = {i: v for i, v in pub.items() if i in prov}
    labels = {}
    for l in open(d / "human_labels.jsonl"):
        r = json.loads(l)
        if r["annotator"] == args.annotator:
            labels[r["item_id"]] = r["label"]
    groups = {"annoying": [], "legitimate": []}
    for i, it in pub.items():
        if prov[i]["corpus"] in args.corpora and "repeat_of" not in prov[i] and labels.get(i) in groups:
            groups[labels[i]].append(it)

    prior_recs = [r for r in map(json.loads, open(args.prior_instances)) if r["pattern_id"] == "rather_than"]

    lines = [f"# Lexical contrast: annoying vs. legitimate (arXiv 2026, annotator {args.annotator})\n",
             f"{len(groups['annoying'])} annoying vs. {len(groups['legitimate'])} legitimate items; "
             f"prior from {len(prior_recs)} dataset-(2) instances. Terms in >= {args.min_items} items.\n"]
    scored = []
    for win in ("sentence", "context"):
        item_freq = {g: collections.Counter() for g in groups}
        texts = {g: {} for g in groups}
        for g, items in groups.items():
            for it in items:
                text = windows(it)[win]
                grams = ngrams(tokens(text))
                item_freq[g].update(set(grams))
                texts[g][it["item_id"]] = text
        prior_freq = collections.Counter()
        for r in prior_recs:
            prior_freq.update(set(ngrams(tokens(windows(r)[win]))))
        keep = [w for w in prior_freq if item_freq["annoying"][w] + item_freq["legitimate"][w] >= args.min_items]
        prior_p = {w: min(prior_freq[w] / len(prior_recs), 0.999) for w in keep}
        z = log_odds(item_freq["annoying"], len(groups["annoying"]), item_freq["legitimate"],
                     len(groups["legitimate"]), prior_p, args.prior_strength)
        scored.append((win, z, item_freq, texts))
    allp = [(win, w, math.erfc(abs(v) / math.sqrt(2))) for win, z, _, _ in scored for w, v in z.items()]
    order = sorted(range(len(allp)), key=lambda k: allp[k][2])
    q, prev = [0.0] * len(allp), 1.0
    for rank, k in reversed(list(enumerate(order, 1))):
        prev = min(prev, allp[k][2] * len(allp) / rank)
        q[k] = prev
    qv = {(win, w): q[k] for k, (win, w, _) in enumerate(allp)}
    lines.append(f"Benjamini-Hochberg over {len(allp)} terms (sentence and context): "
                 f"{sum(v < .05 for v in q)} with q < .05, {sum(v < .10 for v in q)} with q < .10.\n")
    for win, z, item_freq, texts in scored:
        ranked = sorted(z, key=z.get, reverse=True)
        for direction, terms in (("annoying", ranked[:args.top]), ("legitimate", ranked[::-1][:args.top])):
            lines.append(f"\n## {win}: terms most typical of {direction} items\n")
            lines.append("| term | z | q | annoying items | legitimate items | example (annoying item) |")
            lines.append("|---|---|---|---|---|---|")
            for w in terms:
                ex = next((t for t in texts["annoying"].values() if re.search(r"(?<![a-z0-9])" + re.escape(w).replace(r"\ ", r"\s*") + r"(?![a-z0-9])", t.lower())), "")
                ex = snippet(ex, w).replace("|", "\\|") if ex else ""
                lines.append(f"| `{w}` | {z[w]:.2f} | {qv[(win, w)]:.2f} | {item_freq['annoying'][w]}/{len(groups['annoying'])} | "
                             f"{item_freq['legitimate'][w]}/{len(groups['legitimate'])} | {ex} |")
    report = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()

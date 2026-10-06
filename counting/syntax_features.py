#!/usr/bin/env python3
r"""
Syntactic features of annoying vs. legitimate "rather than" items (arXiv
2026 strata; ACL 2019 has no annoying items), parsed with spaCy.

Per item, from the dependency parse of the sentence. X and Y are read off
the tree: X is the head "than" attaches to; Y is the complement of "than"
(pobj/pcomp/...), or, when "than" is parsed as a coordinator, the conjunct of
X that follows it, or failing both, the highest word after "than" in the
same clause. Spans are the heads' subtrees (X without the "than" and
Y material). With --xy-source llm, X and Y come from extract_xy.py instead.
  initial           "rather than" opens the sentence ("Rather than Y, X")
  cat_X, cat_Y      phrase type of X and Y from the POS of the span's head:
                    NP (noun/pronoun), VP-ing (gerund), VP (other verb),
                    AdjP, PP, other
  parallel          X and Y have the same phrase type
  len_X, len_Y      span length in tokens
  subj              subject of the main clause: we / our-NP (possessive
                    "our ...") / other pronoun / NP / none
  passive           the main clause is passive
  modal             the sentence contains a modal verb
  negation          the sentence contains a negation dependency
  sent_len          sentence length in tokens
  depth             maximum dependency-tree depth

Binary features are compared with Fisher's exact test, numeric ones with
Mann-Whitney; Benjamini-Hochberg q-values are computed across all tests.
With --stratify (items from several corpora, e.g. arXiv and GPT items of
batch 3), the tests control for corpus: Cochran-Mantel-Haenszel for binary
features, and for numeric ones a permutation test of the Mann-Whitney U
statistic with labels shuffled within each corpus.

Usage:
    python3 syntax_features.py ../data/annotation [--xy-source tree|llm] \
        [--xy ../data/annotation/xy_alternatives.jsonl] [--annotator A1] [--model en_core_web_sm] [--out features.jsonl]
"""
import argparse
import collections
import json
import re
import statistics
from pathlib import Path

import spacy
from scipy import stats

ARXIV2026 = {"arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"}


def locate(sentence, phrase):
    pat = r"\s+".join(re.escape(w) for w in phrase.split())
    m = re.search(pat, sentence)
    return (m.start(), m.end()) if m else (None, None)


def span_tokens(doc, start, end):
    return [t for t in doc if t.idx < end and t.idx + len(t) > start and not t.is_space]


def head_of(tokens):
    ids = {t.i for t in tokens}
    heads = [t for t in tokens if t.head.i not in ids or t.head.i == t.i]
    return heads[0] if heads else None


def phrase_type(tokens):
    h = head_of(tokens)
    if h is None:
        return "other"
    if h.pos_ in ("NOUN", "PROPN", "PRON", "NUM"):
        return "NP"
    if h.pos_ in ("VERB", "AUX"):
        return "VP-ing" if h.tag_ == "VBG" else "VP"
    if h.pos_ == "ADJ":
        return "AdjP"
    if h.pos_ == "ADP":
        return "PP"
    return "other"


def depth(tok):
    d = 0
    while tok.head.i != tok.i:
        tok, d = tok.head, d + 1
    return d


def main_clause(doc, rt_start):
    rt = next((t for t in doc if t.idx >= rt_start), doc[0])
    return rt.sent.root


def subject_type(root):
    subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass", "csubj")]
    if not subj:
        return "none"
    s = subj[0]
    low = s.text.lower()
    if low == "we":
        return "we"
    if any(c.lower_ == "our" and c.dep_ == "poss" for c in s.children):
        return "our-NP"
    if s.pos_ == "PRON":
        return "other pronoun"
    return "NP"


def tree_xy(doc, rt_start):
    """X and Y token lists from the parse, or (None, None) if "than" is not found."""
    than = next((t for t in doc if t.lower_ == "than" and t.idx >= rt_start), None)
    if than is None:
        return None, None
    x_head = than.head if than.head.i != than.i else None
    comp = [c for c in than.children if c.lower_ != "rather" and not c.is_punct]
    if comp:
        y_head = comp[0]
    elif x_head is not None:
        conj = [c for c in x_head.children if c.dep_ == "conj" and c.i > than.i]
        y_head = conj[0] if conj else None
    else:
        y_head = None
    if y_head is None:
        # fallback: the highest word after "than" in the same clause
        # (the parser sometimes attaches the second conjunct higher up)
        after = []
        for t in doc[than.i + 1:]:
            if t.text in (";", ".", ":") or (t.dep_ == "cc" and t.i > than.i + 1):
                break
            if not t.is_punct:
                after.append(t)
        if after:
            y_head = min(after, key=depth)
            if x_head is None:
                x_head = than.head
    if y_head is None or x_head is None:
        return None, None
    y = list(y_head.subtree)
    drop = {t.i for t in than.subtree} | {t.i for t in y}
    x = [t for t in x_head.subtree if t.i not in drop and not (t.dep_ == "conj" and t.i > than.i)]
    return x, y


def features(nlp, it, xy, source):
    s = it["sentence"]
    doc = nlp(s)
    f = {}
    lead = s[:it["local_start"]]
    f["initial"] = not re.search(r"[A-Za-z0-9]", lead)
    if source == "tree":
        x, y = tree_xy(doc, it["local_start"])
        spans = {"x": x or [], "y": y or []}
    else:
        spans = {}
        for name in ("x", "y"):
            a, b = locate(s, xy[name])
            spans[name] = span_tokens(doc, a, b) if a is not None else []
    for name, toks in spans.items():
        f[f"cat_{name.upper()}"] = phrase_type(toks) if toks else "other"
        f[f"len_{name.upper()}"] = len(toks)
    f["parallel"] = f["cat_X"] == f["cat_Y"]
    root = main_clause(doc, it["local_start"])
    f["subj"] = subject_type(root)
    f["passive"] = any(c.dep_ in ("nsubjpass", "auxpass") for c in root.children)
    f["modal"] = any(t.tag_ == "MD" for t in doc)
    f["negation"] = any(t.dep_ == "neg" for t in doc)
    f["sent_len"] = sum(1 for t in doc if not t.is_space and not t.is_punct)
    f["depth"] = max((depth(t) for t in doc), default=0)
    return f


def bh(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    q, prev = [0.0] * len(pvals), 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, pvals[i] * len(pvals) / rank)
        q[i] = prev
    return q


def cmh_p(a_has, groups):
    """Cochran-Mantel-Haenszel test (continuity-corrected) over strata; a_has: (stratum, annoying, has) rows."""
    num = var = 0.0
    for st in set(g for g, _, _ in a_has):
        rows = [(ann, has) for g, ann, has in a_has if g == st]
        n = len(rows)
        if n < 2:
            continue
        a = sum(ann and has for ann, has in rows)
        r1 = sum(ann for ann, _ in rows); c1 = sum(has for _, has in rows)
        num += a - r1 * c1 / n
        var += r1 * (n - r1) * c1 * (n - c1) / (n * n * (n - 1))
    if var == 0:
        return 1.0
    return stats.chi2.sf((abs(num) - 0.5) ** 2 / var, 1)


def strat_perm_p(vals, perms=4000, seed=7):
    """Two-sided permutation p of Mann-Whitney U, labels shuffled within strata; vals: (stratum, annoying, value)."""
    import random
    rng = random.Random(seed)
    x = [v for _, _, v in vals]
    ranks = stats.rankdata(x)
    strata = collections.defaultdict(list)
    for k, (g, ann, _) in enumerate(vals):
        strata[g].append(k)
    lab = [ann for _, ann, _ in vals]
    n1 = sum(lab); n2 = len(lab) - n1
    def u(labels):
        return sum(r for r, l in zip(ranks, labels) if l) - n1 * (n1 + 1) / 2
    centre = n1 * n2 / 2
    obs = abs(u(lab) - centre)
    hits = 0
    for _ in range(perms):
        perm = lab[:]
        for idx in strata.values():
            sub = [perm[k] for k in idx]
            rng.shuffle(sub)
            for k, v in zip(idx, sub):
                perm[k] = v
        hits += abs(u(perm) - centre) >= obs
    return (hits + 1) / (perms + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--corpora", nargs="+", default=sorted(ARXIV2026), help="corpora whose items are compared")
    ap.add_argument("--xy", help="xy_alternatives.jsonl, needed only with --xy-source llm")
    ap.add_argument("--xy-source", choices=["tree", "llm"], default="tree")
    ap.add_argument("--annotator", default="A1")
    ap.add_argument("--model", default="en_core_web_sm")
    ap.add_argument("--out")
    ap.add_argument("--stratify", action="store_true", help="control for corpus (CMH; within-corpus permutation)")
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    nlp = spacy.load(args.model)

    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl")) if "excluded" not in r}
    pub = {json.loads(l)["item_id"]: json.loads(l) for l in open(d / "items_public.jsonl")}
    pub = {i: v for i, v in pub.items() if i in prov}
    xy = {json.loads(l)["item_id"]: json.loads(l) for l in open(args.xy)} if args.xy else {}
    labels = {}
    for l in open(d / "human_labels.jsonl"):
        r = json.loads(l)
        if r["annotator"] == args.annotator:
            labels[r["item_id"]] = r["label"]
    items = sorted(i for i in pub if prov[i]["corpus"] in args.corpora and "repeat_of" not in prov[i]
                   and labels.get(i) in ("annoying", "legitimate") and (args.xy_source == "tree" or i in xy))
    rows = [dict(item_id=i, label=labels[i], stratum=prov[i]["corpus"], **features(nlp, pub[i], xy.get(i), args.xy_source))
            for i in items]
    missing = sum(r["cat_Y"] == "other" and r["len_Y"] == 0 for r in rows)
    ann = [r for r in rows if r["label"] == "annoying"]
    leg = [r for r in rows if r["label"] == "legitimate"]

    tests = []  # (feature, value-or-stat, ann summary, leg summary, p)
    for feat in ("initial", "parallel", "passive", "modal", "negation"):
        a, b = sum(r[feat] for r in ann), sum(r[feat] for r in leg)
        p = (cmh_p([(r["stratum"], r["label"] == "annoying", bool(r[feat])) for r in rows], None) if args.stratify
             else stats.fisher_exact([[a, len(ann) - a], [b, len(leg) - b]])[1])
        tests.append((feat, "", f"{a}/{len(ann)} ({a / len(ann):.0%})", f"{b}/{len(leg)} ({b / len(leg):.0%})", p))
    for feat in ("cat_X", "cat_Y", "subj"):
        for val in sorted({r[feat] for r in rows}):
            a, b = sum(r[feat] == val for r in ann), sum(r[feat] == val for r in leg)
            if a + b < 5:
                continue
            p = (cmh_p([(r["stratum"], r["label"] == "annoying", r[feat] == val) for r in rows], None) if args.stratify
                 else stats.fisher_exact([[a, len(ann) - a], [b, len(leg) - b]])[1])
            tests.append((feat, val, f"{a}/{len(ann)} ({a / len(ann):.0%})", f"{b}/{len(leg)} ({b / len(leg):.0%})", p))
    for feat in ("len_X", "len_Y", "sent_len", "depth"):
        a, b = [r[feat] for r in ann], [r[feat] for r in leg]
        p = (strat_perm_p([(r["stratum"], r["label"] == "annoying", r[feat]) for r in rows]) if args.stratify
             else stats.mannwhitneyu(a, b).pvalue)
        tests.append((feat, "median", f"{statistics.median(a):g}", f"{statistics.median(b):g}", p))
    qs = bh([t[4] for t in tests])
    print(f"items ({", ".join(args.corpora)}): {len(ann)} annoying, {len(leg)} legitimate; parser {args.model}; "
          f"X/Y from {args.xy_source}; no X/Y found for {missing} items\n")
    print(f"{'feature':10s} {'value':14s} {'annoying':>14s} {'legitimate':>16s} {'p':>7s} {'q':>7s}")
    for (feat, val, a, b, p), q in zip(tests, qs):
        print(f"{feat:10s} {val:14s} {a:>14s} {b:>16s} {p:7.3f} {q:7.3f}")
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

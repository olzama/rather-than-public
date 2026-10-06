#!/usr/bin/env python3
r"""
Passages for the authorship-guess task: annotators read short passages from
the papers behind annotation-pool items, with nothing highlighted, and guess
whether each was written by a person or by an LLM. The guesses are then
related to the same annotators' annoyance labels for items from the same
papers.

For each selected pool item, one passage of --sentences consecutive
sentences from the item's paper (full text, text/):
  vicinity (most passages): from just before or just after the context the
    annotator saw (the item's sentence and its neighbours), not overlapping it;
  seen (--seen-share of passages): a window containing the item's sentence
    (preferably with its two neighbours, the text the annotator saw before).
Passages may or may not contain "rather than". Sentence boundaries follow
extract_antithesis_instances.py's heuristic.

Selection: pool items that both annotators labeled (neither garbled): all
arXiv 2026 items either labeled annoying, plus random legitimate arXiv 2026
items and random ACL 2019 items up to --n.

Text is normalized so the two extraction methods look alike: hyphenation
and ligatures fixed, page footers of the proceedings removed, whitespace collapsed, author-year and numeric citations
in the ACL 2019 PDF text replaced by "[redacted]" (the arXiv text already has
it). Heading lines are dropped. Passages with little prose (digits, symbols),
outside 40-160 words, or with extraction gaps ("( )", a space before a
comma) are skipped in favour of the other side, then the next window.

Output: <out_dir>/passages_public.jsonl (pid, text) and
passages_provenance.jsonl (pid, item_id, kind, side, corpus, doc_id).

Usage:
    python3 authorship_passages.py ../data/annotation ../../data ../data/authorship \
        [--annotators A1 A2] [--n 150] [--seen-share 0.25] [--sentences 3] [--seed 20260928]
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

TEXT_DIR = {"acl2019": "acl2019/text", "arxiv2026": "arxiv2026/text", "high_count_arxiv2026": "arxiv2026/text",
            "arxiv_v1": "arxiv_versions/text", "arxiv_latest": "arxiv_versions/text"}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
CITE = re.compile(r"\((?:[^()]*?\b(?:19|20)\d\d[a-z]?)(?:;[^()]*?\b(?:19|20)\d\d[a-z]?)*\)"
                  r"|\b[A-Z][A-Za-z'\-]+(?: et al\.| and [A-Z][A-Za-z'\-]+)? \((?:19|20)\d\d[a-z]?\)"
                  r"|\[\d+(?:[,–\-]\s*\d+)*\]")
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def drop_headings(text):
    """Remove heading lines: at most 8 words, no final punctuation, next line capitalized."""
    lines = text.split("\n")
    keep = []
    for j, line in enumerate(lines):
        t = line.strip()
        nxt = next((x.strip() for x in lines[j + 1:] if x.strip()), "")
        if t and len(t.split()) <= 8 and not re.search(r"[.!?:;,]$", t) and nxt[:1].isupper():
            continue
        keep.append(line)
    return "\n".join(keep)


FOOTER = re.compile(r"(\w?)\s*\d*\s*(?:Proceedings of [^\n]{0,400}?|,?\s*pages \d+[\u2013-]\d+ [^\n]{0,160}?)"
                    r"Association for Computational Linguistics\s*(\w?)")


def drop_footer(m):
    """Remove a PDF page footer; rejoin a word it split ("con28 Proceedings ... struction")."""
    if m.group(1) and m.group(2) and not m.group(0)[1:2].isspace():
        return m.group(1) + m.group(2)
    return m.group(1) + " " + m.group(2)


def normalize(text, pdf):
    for a, b in LIGATURES.items():
        text = text.replace(a, b)
    text = drop_headings(text)
    if pdf:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = CITE.sub("[redacted]", text)
        text = FOOTER.sub(drop_footer, " ".join(text.split()))
    return " ".join(text.split())


def good(sents):
    t = " ".join(sents)
    words = t.split()
    letters = sum(c.isalpha() for c in t)
    gaps = re.search(r"\(\s*\)|\s[,.;:]|^\W|\(\s*[,;]", t)
    return (40 <= len(words) <= 160 and letters > 0.7 * len(t.replace(" ", "")) and t.count("[redacted]") <= 4
            and not gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("data_dir", help="root with acl2019/text, arxiv2026/text, arxiv_versions/text")
    ap.add_argument("out_dir")
    ap.add_argument("--annotators", nargs=2, default=["A1", "A2"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--acl", type=int, default=15)
    ap.add_argument("--seen-share", type=float, default=0.25)
    ap.add_argument("--sentences", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260928)
    args = ap.parse_args()
    d, root, out = Path(args.annotation_dir), Path(args.data_dir), Path(args.out_dir)
    rng = random.Random(args.seed)
    prov = {r["item_id"]: r for r in map(json.loads, open(d / "items_provenance.jsonl"))
            if "repeat_of" not in r and "excluded" not in r}
    pub = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl"))}
    labels = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        labels[r["annotator"]][r["item_id"]] = r["label"]
    a, b = args.annotators
    both = sorted(i for i in pub if i in prov and i in labels[a] and i in labels[b]
                  and "garbled" not in (labels[a][i], labels[b][i]))
    arx = [i for i in both if prov[i]["corpus"] != "acl2019"]
    annoying = [i for i in arx if "annoying" in (labels[a][i], labels[b][i])]
    legit = [i for i in arx if i not in annoying]
    acl = [i for i in both if prov[i]["corpus"] == "acl2019"]
    n_legit = max(0, args.n - len(annoying) - args.acl)
    chosen = annoying + rng.sample(legit, n_legit) + rng.sample(acl, args.acl)
    rng.shuffle(chosen)
    seen = set(rng.sample(chosen, round(args.seen_share * len(chosen))))

    texts = {}
    public, provenance, skipped = [], [], collections.Counter()
    k = args.sentences
    for i in chosen:
        p = prov[i]
        key = (p["corpus"], p["doc_id"])
        if key not in texts:
            raw = (root / TEXT_DIR[p["corpus"]] / f"{p['doc_id']}.txt").read_text(errors="replace")
            texts[key] = SENT_SPLIT.split(normalize(raw, p["corpus"] == "acl2019"))
        sents = texts[key]
        target = normalize(pub[i]["sentence"], p["corpus"] == "acl2019")
        idx = next((j for j, s in enumerate(sents) if target[:60] in s or s[:60] in target), None)
        if idx is None:
            skipped["item sentence not found"] += 1
            continue
        if i in seen:
            windows = [("seen", "", max(0, idx - 1)), ("seen", "", idx), ("seen", "", max(0, idx - 2))]
        else:
            sides = ["before", "after"]
            rng.shuffle(sides)
            windows = []
            for shift in range(0, 4):
                for side in sides:
                    start = idx - 2 - k - shift if side == "before" else idx + 2 + shift
                    windows.append(("vicinity", side, start))
        pick = None
        for kind, side, start in windows:
            if start < 0 or start + k > len(sents):
                continue
            if good(sents[start:start + k]):
                pick = (kind, side, start)
                break
        if pick is None:
            skipped["no usable window"] += 1
            continue
        kind, side, start = pick
        pid = f"auth_{len(public) + 1:04d}"
        public.append({"pid": pid, "text": " ".join(sents[start:start + k])})
        provenance.append({"pid": pid, "item_id": i, "kind": kind, "side": side, "corpus": p["corpus"],
                           "doc_id": p["doc_id"]})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "passages_public.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in public)
    with open(out / "passages_provenance.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in provenance)
    kinds = collections.Counter(r["kind"] for r in provenance)
    rt = sum(bool(re.search(r"\brather than\b", r["text"], re.I)) for r in public)
    print(f"{len(public)} passages ({dict(kinds)}); {rt} contain 'rather than'; skipped: {dict(skipped)}; "
          f"selected from {len(annoying)} annoying-to-either, {len(legit)} legitimate arXiv, {len(acl)} ACL items")


if __name__ == "__main__":
    main()

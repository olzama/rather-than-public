#!/usr/bin/env python3
r"""
Passages for the authorship-guess task on the annotation pool: pool items,
shown as the annotators saw them (context before, the sentence, context
after) with nothing highlighted, to be judged on a hunch as written by a
person or by an LLM, mixed with filler passages that contain no antithesis
construction.

Items: every pool item that either annotator (--annotators) labeled
annoying, and a random --legit-share of the other pool items (or, with
--legit-count and --acl-count, that many other arXiv 2026 and ACL 2019 items); excluded items
and hidden repeats are left out, and so are, with --exclude-labeled-by NAME,
the items NAME labeled for annoyance, and with --exclude-strawman, every item
of the straw-man coding sets. With --extra DIR --extra-corpora C ..., all
items of another pool folder (e.g. annotation batch 3) from the listed
corpora are added (with --extra-per-corpus N, N per corpus), e.g. GPT items
(known LLM-written) and ACL 2019 items (known human-written) as a calibration
set.

Fillers: one per --filler-ratio items, each from the paper of a randomly
chosen item: --sentences consecutive sentences that match none of the
antithesis patterns (antithesis_patterns.py) and lie at least three sentences
away from the item's sentence. Fillers keep the construction from being the
one thing all passages share, and they allow a direct test of whether it
raises the "LLM" hunch.

Text is normalized as in authorship_passages.py (ligatures, hyphenation,
citations and page footers in the ACL 2019 PDF text, whitespace). Item
passages over 300 words, all broken extractions, are cut to the sentence, or
to 300 words around the match.

Output, in the format of authorship_passages.py so that
build_authorship_page.py and ../counting/authorship_analysis.py read it:
<out_dir>/passages_public.jsonl (pid, text) and passages_provenance.jsonl
(pid, item_id, kind "seen" or "filler", corpus, doc_id). Pids are
"pool_<item_id>" and "fill_<n>", so the answers never collide with those of
the passage task.

Usage:
    python3 pool_authorship_passages.py ../data/annotation ../../data ../data/authorship_pool \
        --exclude-labeled-by A5 --exclude-strawman \
        --extra ../data/annotation_b3 --extra-corpora llm_gpt-4o llm_gpt-5.6-sol llm_gpt-6-sol acl2019 \
        --llm-text-dir <llm_instances.py output>
"""
import argparse
import json
import random
import re
from pathlib import Path

from antithesis_patterns import PATTERNS
from authorship_passages import CITE, FOOTER, LIGATURES, SENT_SPLIT, TEXT_DIR, drop_footer, good, normalize

MAX_WORDS = 300
ANTITHESIS = [re.compile(p, re.I) for _pid, _tier, _label, p, _note in PATTERNS]


def clean(text, pdf):
    for a, b in LIGATURES.items():
        text = text.replace(a, b)
    if pdf:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = CITE.sub("[redacted]", text)
        text = FOOTER.sub(drop_footer, " ".join(text.split()))
    return " ".join(text.split())


def item_text(r, pdf):
    text = " ".join(clean(r[k], pdf) for k in ("context_before", "sentence", "context_after") if r.get(k))
    if len(text.split()) > MAX_WORDS:  # broken extractions: keep the sentence, or a window around the match
        text = clean(r["sentence"], pdf)
        if len(text.split()) > MAX_WORDS:
            pos = len(clean(r["sentence"][:r["local_start"]], pdf).split())
            words = text.split()
            lo = max(0, pos - MAX_WORDS // 2)
            text = " ".join(words[lo:lo + MAX_WORDS])
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("data_dir", help="root with acl2019/text, arxiv2026/text, arxiv_versions/text")
    ap.add_argument("out_dir")
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    ap.add_argument("--legit-share", type=float, default=0.5)
    ap.add_argument("--legit-count", type=int, help="other arXiv 2026 items to draw (instead of --legit-share)")
    ap.add_argument("--acl-count", type=int, help="other ACL 2019 items to draw (with --legit-count)")
    ap.add_argument("--extra-per-corpus", type=int, help="items to draw from each --extra-corpora corpus")
    ap.add_argument("--exclude-labeled-by", nargs="*", default=[])
    ap.add_argument("--exclude-strawman", action="store_true")
    ap.add_argument("--extra")
    ap.add_argument("--extra-corpora", nargs="*", default=[])
    ap.add_argument("--llm-text-dir", help="llm_instances.py output (<model>/<doc_id>.txt)")
    ap.add_argument("--filler-ratio", type=float, default=2.0, help="items per filler")
    ap.add_argument("--sentences", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    d, root, out = Path(args.annotation_dir), Path(args.data_dir), Path(args.out_dir)

    skip, annoying = set(), set()
    for r in map(json.loads, open(d / "human_labels.jsonl")):
        if r["annotator"] in args.exclude_labeled_by:
            skip.add(r["item_id"].removeprefix("rep__"))
        if r["annotator"] in args.annotators and r["label"] == "annoying":
            annoying.add(r["item_id"])
    if args.exclude_strawman:
        for f in (d / "strawman" / "dev.jsonl", d / "strawman" / "test.jsonl"):
            skip |= {json.loads(l)["item_id"] for l in open(f)}

    items = []  # (item_id, public row, provenance row)
    sources = [(d, None)] + ([(Path(args.extra), set(args.extra_corpora))] if args.extra else [])
    for src, corpora in sources:
        prov = {r["item_id"]: r for r in map(json.loads, open(src / "items_provenance.jsonl"))}
        rows = [r for r in map(json.loads, open(src / "items_public.jsonl"))
                if not prov[r["item_id"]].get("excluded") and "repeat_of" not in prov[r["item_id"]]
                and r["item_id"] not in skip and (corpora is None or prov[r["item_id"]]["corpus"] in corpora)]
        if corpora is None:  # main pool: all annoying items, and a share or a count of the rest
            ann = [r for r in rows if r["item_id"] in annoying]
            rest = [r for r in rows if r["item_id"] not in annoying]
            if args.legit_count is not None:
                arx = [r for r in rest if prov[r["item_id"]]["corpus"] != "acl2019"]
                acl = [r for r in rest if prov[r["item_id"]]["corpus"] == "acl2019"]
                rows = ann + rng.sample(arx, args.legit_count) + rng.sample(acl, args.acl_count or 0)
            else:
                rows = ann + rng.sample(rest, round(args.legit_share * len(rest)))
        elif args.extra_per_corpus is not None:
            by = {}
            for r in rows:
                by.setdefault(prov[r["item_id"]]["corpus"], []).append(r)
            rows = [r for c in sorted(by) for r in rng.sample(by[c], min(args.extra_per_corpus, len(by[c])))]
        items += [(r["item_id"], r, prov[r["item_id"]]) for r in rows]

    def full_text(p):
        if p["corpus"].startswith("llm_"):
            path = Path(args.llm_text_dir) / p["corpus"][4:] / f"{p['doc_id']}.txt"
        else:
            path = root / TEXT_DIR[p["corpus"]] / f"{p['doc_id']}.txt"
        return SENT_SPLIT.split(normalize(path.read_text(errors="replace"), p["corpus"] == "acl2019"))

    fillers, tried = [], 0
    pool = items[:]
    rng.shuffle(pool)
    n_fill = round(len(items) / args.filler_ratio)
    used = set()
    k = args.sentences
    for item_id, r, p in pool:
        if len(fillers) >= n_fill:
            break
        tried += 1
        sents = full_text(p)
        target = clean(r["sentence"], p["corpus"] == "acl2019")
        idx = next((j for j, s in enumerate(sents) if target[:60] in s or s[:60] in target), None)
        if idx is None:
            continue
        starts = [s for s in range(len(sents) - k + 1) if s + k <= idx - 3 or s >= idx + 4]
        rng.shuffle(starts)
        for s in starts:
            window = sents[s:s + k]
            text = " ".join(window)
            if (p["doc_id"], s) in used or not good(window) or any(rx.search(text) for rx in ANTITHESIS):
                continue
            used.add((p["doc_id"], s))
            fillers.append((item_id, text, p))
            break

    passages = [("pool_" + i, item_text(r, p["corpus"] == "acl2019"), i, "seen", p) for i, r, p in items]
    passages += [(f"fill_{n + 1:04d}", text, i, "filler", p) for n, (i, text, p) in enumerate(fillers)]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "passages_public.jsonl", "w") as pub, open(out / "passages_provenance.jsonl", "w") as pv:
        for pid, text, item_id, kind, p in passages:
            pub.write(json.dumps({"pid": pid, "text": text}) + "\n")
            pv.write(json.dumps({"pid": pid, "item_id": item_id, "kind": kind, "corpus": p["corpus"],
                                 "doc_id": p["doc_id"]}) + "\n")
    print(f"wrote {len(items)} items and {len(fillers)} fillers (from {tried} papers tried) to {out}")


if __name__ == "__main__":
    main()

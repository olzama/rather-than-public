#!/usr/bin/env python3
r"""
Flag documents that are not usable English text, for exclusion from all
rates and samples.

Each document is split into prose chunks (paragraphs of at least 30
letter-words; lines dominated by digits or symbols are skipped). Up to
--max-chunks chunks, spread evenly over the document, are passed to
langdetect, and the English share is the fraction of their characters
detected as English. A document is flagged
  no_text      fewer than --min-words letter-words in total
  non_english  English share below --min-english
  non_english  CJK characters (Chinese, Japanese, Korean) divided by 1.5
               (about one word per 1.5 characters) exceed the number of
               Latin-alphabet words: such paragraphs have too few
               space-separated words to be chunked, so this is checked
               directly
Mixed documents (e.g. a Chinese paper with an English abstract) are thus
judged by their dominant language.

Output: TSV with corpus, doc_id, reason, english_share, words, top
non-English language; one row per flagged document.

Usage:
    python3 language_filter.py --corpus acl2019=../../data/acl2019/text_body \
        --corpus arxiv2026=../../data/arxiv2026/text_body \
        --corpus arxiv_versions=../../data/arxiv_versions/text_body \
        --out ../data/excluded_documents.tsv
"""
import argparse
import collections
import re
from pathlib import Path

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0
WORD_RE = re.compile(r"[^\W\d_]{2,}")
LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def chunks(text, min_words=30):
    out = []
    for para in re.split(r"\n\s*\n", text):
        words = WORD_RE.findall(para)
        if len(words) >= min_words and sum(len(w) for w in words) > 0.5 * len(para.strip()):
            out.append(para)
    return out


def assess(text, max_chunks, min_words, min_english):
    n_words = len(WORD_RE.findall(text))
    if n_words < min_words:
        return "no_text", 0.0, n_words, ""
    if len(CJK_RE.findall(text)) / 1.5 > len(LATIN_WORD_RE.findall(text)):
        return "non_english", 0.0, n_words, "cjk"
    cs = chunks(text)
    if len(cs) > max_chunks:
        step = len(cs) / max_chunks
        cs = [cs[int(i * step)] for i in range(max_chunks)]
    by_lang = collections.Counter()
    for c in cs:
        try:
            by_lang[detect(c)] += len(c)
        except LangDetectException:
            pass
    total = sum(by_lang.values())
    if not total:
        return "no_text", 0.0, n_words, ""
    share = by_lang["en"] / total
    other = next((l for l, _ in by_lang.most_common() if l != "en"), "")
    return ("non_english" if share < min_english else None), share, n_words, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chunks", type=int, default=30)
    ap.add_argument("--min-words", type=int, default=200)
    ap.add_argument("--min-english", type=float, default=0.5)
    args = ap.parse_args()
    rows = []
    for spec in args.corpus:
        name, d = spec.split("=", 1)
        files = sorted(Path(d).glob("*.txt"))
        flagged = 0
        for f in files:
            reason, share, n, other = assess(f.read_text(errors="replace"), args.max_chunks, args.min_words, args.min_english)
            if reason:
                rows.append((name, f.stem, reason, f"{share:.2f}", str(n), other))
                flagged += 1
        print(f"{name}: {flagged} of {len(files)} flagged")
    with open(args.out, "w") as f:
        f.write("corpus\tdoc_id\treason\tenglish_share\twords\tother_language\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Extract every individual match of the antithesis-family patterns (see
antithesis_patterns.py) from a text corpus, with enough provenance to
find it again: which document, which character span, and the sentence
it occurred in. This is the per-instance companion to
count_antithesis_patterns.py's aggregate counts -- for manual review,
qualitative spot-checks, or a future legitimate-vs-annoying annotation
pass (the paper's stated goal (b)).

Sentence boundaries are a simple regex heuristic (split before a
capital/digit that follows [.!?] and whitespace), not a real sentence
segmenter -- it will occasionally misfire on abbreviations, citations
("et al."), and decimals. Good enough to give a reviewer usable context,
not guaranteed to be exactly one grammatical sentence.

Usage:
    python3 extract_antithesis_instances.py <text_dir> <out.jsonl> \
        [--glob "*.txt"] [--corpus-name NAME]

Writes one JSON record per match:
    {"corpus": ..., "doc_id": ..., "pattern_id": ..., "tier": ...,
     "label": ..., "match_text": ..., "char_start": ..., "char_end": ...,
     "sentence": ..., "local_start": ..., "local_end": ...,
     "context_before": ..., "context_after": ...}

local_start/local_end are the match's offset within "sentence" itself
(as opposed to char_start/char_end, which are offsets into the whole
document) -- needed because a sentence can contain more than one match
of the same or a different pattern; each match gets its own record, and
local_start/local_end is how a consumer (e.g. the annotation UI) knows
*which* occurrence a given record is about, rather than highlighting
every occurrence in the sentence indiscriminately.

context_before/context_after are the immediately preceding/following
heuristic sentences (empty string at a document boundary) -- for
display only, not part of any dedup key, so they can be extended or
changed without touching identity.

doc_id is the filename stem (e.g. an ACL Anthology id, or an arXiv id
optionally suffixed "_v1"/"_v2" for the version-diff corpora).
"""
import argparse
import json
import re
import sys
from pathlib import Path

from antithesis_patterns import compile_patterns, pattern_meta

SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def sentence_spans(text):
    """Return a sorted list of (start, end) character spans, one per
    heuristic sentence, covering the whole text contiguously."""
    spans = []
    start = 0
    for m in SENT_BOUNDARY_RE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def sentence_for_span(spans, start, end):
    """Find the sentence span containing [start, end), and its index range
    in `spans` (a match spanning multiple heuristic sentences -- shouldn't
    happen given the patterns' same-sentence windows, but be defensive --
    yields more than one covering index). Returns
    (s_start, s_end, first_idx, last_idx); if nothing covers the match,
    first_idx/last_idx are None so callers can skip context lookup."""
    covering_idx = [i for i, s in enumerate(spans) if s[0] <= start < s[1] or s[0] < end <= s[1]]
    if not covering_idx:
        return start, end, None, None
    first_idx, last_idx = covering_idx[0], covering_idx[-1]
    return spans[first_idx][0], spans[last_idx][1], first_idx, last_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text_dir")
    p.add_argument("out_path")
    p.add_argument("--glob", default="*.txt")
    p.add_argument("--corpus-name", default=None, help="label stored in each record; defaults to text_dir's name")
    args = p.parse_args()

    compiled = compile_patterns()
    meta = pattern_meta()
    text_dir = Path(args.text_dir)
    files = sorted(text_dir.glob(args.glob))
    if not files:
        sys.exit(f"no files matching {args.glob} under {text_dir}")

    corpus_name = args.corpus_name or text_dir.name
    total = 0
    with open(args.out_path, "w") as out:
        for f in files:
            text = f.read_text(errors="replace")
            spans = sentence_spans(text)
            doc_id = f.stem
            for pid, rx in compiled.items():
                for m in rx.finditer(text):
                    s_start, s_end, first_idx, last_idx = sentence_for_span(spans, m.start(), m.end())
                    raw_sentence = text[s_start:s_end]
                    sentence = raw_sentence.strip()
                    leading_ws = len(raw_sentence) - len(raw_sentence.lstrip())
                    local_start = m.start() - s_start - leading_ws
                    local_end = m.end() - s_start - leading_ws
                    context_before = ""
                    context_after = ""
                    if first_idx is not None and first_idx > 0:
                        context_before = text[spans[first_idx - 1][0]:spans[first_idx - 1][1]].strip()
                    if last_idx is not None and last_idx + 1 < len(spans):
                        context_after = text[spans[last_idx + 1][0]:spans[last_idx + 1][1]].strip()
                    record = {
                        "corpus": corpus_name,
                        "doc_id": doc_id,
                        "pattern_id": pid,
                        "tier": meta[pid]["tier"],
                        "label": meta[pid]["label"],
                        "match_text": m.group(0),
                        "char_start": m.start(),
                        "char_end": m.end(),
                        "sentence": sentence,
                        "local_start": local_start,
                        "local_end": local_end,
                        "context_before": context_before,
                        "context_after": context_after,
                    }
                    out.write(json.dumps(record) + "\n")
                    total += 1

    print(f"DONE {total} instances across {len(files)} files -> {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

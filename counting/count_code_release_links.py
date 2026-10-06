#!/usr/bin/env python3
r"""
Estimate how many papers link to their OWN code release on GitHub, as
opposed to merely citing a github.com URL for a tool/baseline they used
(e.g. huggingface/transformers, fairseq) -- a grep for any github.com
link alone conflates the two (see paper text: ~52-54% of ACL
2019/arXiv 2026 papers mention github.com at all, most of that citing
other people's tools).

Heuristic: find every github.com URL, then check whether a "code
release" cue phrase (e.g. "our code", "code is available", "we
release", "open-source") appears in the same heuristic sentence or an
immediately adjacent one -- own-code mentions are typically short,
self-contained statements ("Our code is available at
https://github.com/...") or a footnote/caption right next to the URL,
while third-party tool citations name the tool, not "our"/"we release".
Same sentence-boundary heuristic as extract_antithesis_instances.py
(regex, not a real segmenter -- occasionally misfires on abbreviations
and citations).

This is still a heuristic, not ground truth: it will miss own-code
links phrased unusually, and can over-match a nearby cue that isn't
actually about that URL. One specific false-positive shape caught
during development: "We use the implementation available here:
https://github.com/huggingface/neuralcoref" -- a bare "(code/
implementation) available" cue fires even though this is citing a
third-party tool, not releasing the paper's own code. Handled by
requiring either a self-referential cue ("our code", "we release", ...)
or a non-self-referential one ("code is available", ...) with no
citation verb ("we use", "based on", "built on", ...) in the same
window. Treat the numbers as an estimate of the SHAPE of the gap
between "any github.com link" and "own-code link," not exact counts.

Usage:
    python3 count_code_release_links.py <text_dir> [--glob "*.txt"] \
        [--jsonl out.jsonl] [--corpus-name NAME]

Prints, per corpus: total papers, papers with any github.com link, and
papers where a code-release cue appears near one of those links. Pass
--jsonl to also write one record per paper with any github.com link,
including which URL/cue (if any) matched.
"""
import argparse
import json
import re
import sys
from pathlib import Path

URL_RE = re.compile(r"github\.com/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?", re.I)
SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Unambiguous: explicitly self-referential ("our"/"we release"), so a
# nearby citation verb doesn't matter.
STRONG_CUE_RE = re.compile(
    r"\b(?:"
    r"our code|our implementation|our (?:source )?repository|our github|"
    r"official implementation|code for (?:this|our) (?:paper|work)|"
    r"we (?:release|open-?source|publicly release|"
    r"make (?:our|the) (?:code|data and code|code and data) (?:publicly )?available|"
    r"provide (?:our )?code)"
    r")\b",
    re.I,
)

# Ambiguous on their own -- could describe a third-party tool the paper
# used ("the implementation available at ...") rather than the paper's
# own release. Only counted as own-code if CITATION_RE doesn't also
# appear in the same window.
WEAK_CUE_RE = re.compile(
    r"\b(?:"
    r"code (?:is |will be |are |is publicly |is now )*(?:available|released|public)|"
    r"code and data (?:are|is) available|"
    r"source code (?:is )?(?:available|released)|"
    r"implementation (?:is )?(?:available|released|public)|"
    r"code (?:can be found|is hosted|repository)|"
    r"publicly available (?:code|implementation)|"
    r"data and code (?:are|is) available"
    r")\b",
    re.I,
)

# Signals the nearby "available"/"released" cue is about a tool the
# paper is USING or attributing to someone else, not the paper's own
# release.
CITATION_RE = re.compile(
    r"\b(?:we use[ds]?|using the|we utiliz(?:e[ds]?)|we employ(?:ed)?|we adopt(?:ed)?|"
    r"built on|based on the|adapted from|fine-?tuned using|pre-?trained|"
    r"released by|provided by|published by|developed by|"
    r"made available by|open-?sourced by)\b",
    re.I,
)


def clean_url(raw):
    """Extracted text sometimes glues adjacent tokens together with no
    whitespace (a PDF/detex artifact), e.g. ".../STARgithub.com" or
    ".../EmoStancehttps" where a second URL or word immediately follows
    the real one. Truncate at the first sign of that after the
    "github.com/" prefix, and strip trailing punctuation."""
    prefix, rest = raw[:len("github.com/")], raw[len("github.com/"):]
    rest = re.split(r"(?i)(https?|github|www\.)", rest, maxsplit=1)[0]
    return (prefix + rest).rstrip(".").rstrip("/")


def sentence_spans(text):
    spans = []
    start = 0
    for m in SENT_BOUNDARY_RE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def window_for_span(spans, start, end):
    """Character span covering the heuristic sentence containing
    [start, end) plus its immediate neighbors on both sides."""
    covering = [i for i, s in enumerate(spans) if s[0] <= start < s[1] or s[0] < end <= s[1]]
    if not covering:
        return start, end
    lo = max(0, covering[0] - 1)
    hi = min(len(spans) - 1, covering[-1] + 1)
    return spans[lo][0], spans[hi][1]


def check_file(path, corpus_name):
    text = path.read_text(errors="replace")
    urls = list(URL_RE.finditer(text))
    if not urls:
        return None
    spans = sentence_spans(text)
    matched_url, matched_cue, confidence = None, None, None
    for m in urls:
        w_start, w_end = window_for_span(spans, m.start(), m.end())
        window = text[w_start:w_end]
        strong = STRONG_CUE_RE.search(window)
        if strong:
            matched_url, matched_cue, confidence = clean_url(m.group(0)), strong.group(0), "strong"
            break
        weak = WEAK_CUE_RE.search(window)
        if weak and not CITATION_RE.search(window):
            matched_url, matched_cue, confidence = clean_url(m.group(0)), weak.group(0), "weak"
            # keep looking -- a later URL in this file might get a strong match
    return {
        "corpus": corpus_name,
        "doc_id": path.stem,
        "n_github_urls": len(urls),
        "own_code_link": matched_url is not None,
        "confidence": confidence,
        "matched_url": matched_url,
        "matched_cue": matched_cue,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text_dir")
    p.add_argument("--glob", default="*.txt")
    p.add_argument("--jsonl", default=None)
    p.add_argument("--corpus-name", default=None, help="label stored in each record; defaults to text_dir's name")
    args = p.parse_args()

    text_dir = Path(args.text_dir)
    files = sorted(text_dir.glob(args.glob))
    if not files:
        sys.exit(f"no files matching {args.glob} under {text_dir}")

    corpus_name = args.corpus_name or text_dir.name
    records = [r for r in (check_file(f, corpus_name) for f in files) if r is not None]
    total_files = len(files)
    with_any_link = len(records)
    with_own_code = sum(1 for r in records if r["own_code_link"])
    n_strong = sum(1 for r in records if r["confidence"] == "strong")
    n_weak = sum(1 for r in records if r["confidence"] == "weak")

    print(f"corpus: {text_dir}")
    print(f"files: {total_files}")
    print(f"files with >=1 github.com link: {with_any_link} ({100 * with_any_link / total_files:.1f}%)")
    print(f"files with an apparent own-code link: {with_own_code} ({100 * with_own_code / total_files:.1f}%"
          f" of all files, {100 * with_own_code / with_any_link:.1f}% of files with any github.com link)")
    print(f"  of which: strong (self-referential, e.g. \"our code\") = {n_strong}, "
          f"weak (\"code is available\", no nearby citation verb) = {n_weak}")

    if args.jsonl:
        with open(args.jsonl, "w") as out:
            for r in records:
                out.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

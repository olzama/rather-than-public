#!/usr/bin/env python3
r"""
Split each paper's detex'd text into top-level sections (abstract,
introduction, related work, method, ..., references, appendix).

Deliberately conservative: a candidate heading line is only treated as a
section boundary if it matches a whitelist of standard ACL-style section
names (with common aliases -- "Method"/"Approach"/"Methodology" all map
to "method"). Unrecognized standalone-looking lines (subsection/
paragraph headings like "Datasets" or "Copy-Paste.") are left embedded
in whichever recognized section's text they fall under, rather than
creating a new boundary for them -- a wrong boundary silently corrupts
two sections' text, while an unsplit subsection just means slightly
coarser granularity. Run `python3 split_paper_sections.py --report-unmatched
<text_dir>` before trusting this on a new corpus: it prints the most
common candidate headings that AREN'T in the whitelist, which is how you
find out if the alias list needs another entry.

Why this only targets arXiv 2026 (detex'd LaTeX), not ACL 2019
(pdftotext-extracted PDF): checked by hand across several ACL 2019
papers -- pdftotext's column handling decouples a heading's number from
its title by dozens of lines and, worse, scrambles PARAGRAPH READING
ORDER itself (one paper's subsection "3.2" appeared before "3.1" in the
extracted text). That's not a heading-detection problem a better regex
can fix; a heuristic splitter over that text would confidently produce
section spans mixing content from different sections. See
../README.md's Section 5 (or wherever this script's usage note lives)
for the ACL 2019 status.

Heading-candidate structural signal: a line with >=1 blank line before
AND after it, non-empty, no trailing sentence punctuation (rules out
run-in paragraph headings like "Copy-Paste.", which detex renders with
a trailing period; real \section/\subsection titles don't have one in
this corpus), and, after stripping a leading number ("1.", "3.2") for
papers that do retain numbering, case-insensitively equal to a
whitelist alias.

The abstract has no explicit "Abstract" heading in this corpus (0/1841
checked) -- it's typeset by the ACL class's \maketitle machinery, not
literal source text, so detex never sees the word. Everything before
the first recognized heading is "front_matter"; within that, the LAST
paragraph at least --abstract-min-chars long is labeled "abstract" (title/
author/affiliation blocks are short, blank-line-separated fragments;
the abstract is the one long block of prose) and anything before it is
"front_matter" proper. If no paragraph clears the threshold, the whole
preamble stays "front_matter" and there's no "abstract" entry -- check
n_missing_abstract in the summary before assuming full coverage.

Usage:
    python3 split_paper_sections.py <text_dir> [--glob "*.txt"] --jsonl out.jsonl
    python3 split_paper_sections.py <text_dir> --report-unmatched [--top 40]

Writes one JSON record per paper to --jsonl:
    {"doc_id": ..., "sections": [{"category": ..., "heading": <str or null>,
     "text": ...}, ...], "n_recognized_headings": ..., "total_chars": ...,
     "chars_covered": ...}

"chars_covered" always equals "total_chars" (sections partition the
document with no gaps) -- that's a structural guarantee of how this
splits, not a quality claim about where the boundaries fall; it's
printed so a consumer can sanity check nothing was silently dropped by
a bug rather than double-checking it every time.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# category -> aliases (lowercase, whitespace-normalized, no trailing punctuation)
CANONICAL_SECTIONS = {
    "introduction": ["introduction"],
    "related_work": [
        "related work", "related works", "prior work", "background",
        "background and related work", "related work and background",
    ],
    "method": [
        "method", "methods", "methodology", "approach", "approaches",
        "model", "our approach", "our method", "proposed method",
        "proposed approach", "system description", "our model",
    ],
    "preliminaries": [
        "preliminaries", "preliminary", "problem formulation",
        "problem statement", "task formulation", "notation",
    ],
    "experiments": [
        "experiments", "experiment", "experimental setup",
        "experimental settings", "experimental design", "setup",
        "experiment setup", "evaluation setup", "experiments and results",
    ],
    "results": [
        "results", "results and discussion", "results and analysis",
        "experimental results", "findings", "evaluation results",
        "main results", "additional results", "additional experimental results",
    ],
    "analysis": ["analysis", "discussion", "error analysis", "qualitative analysis"],
    "conclusion": [
        "conclusion", "conclusions", "conclusion and future work",
        "conclusions and future work", "summary", "summary and conclusion",
        "discussion and conclusion", "discussion and conclusions",
    ],
    "limitations": ["limitations", "limitation"],
    "ethics": [
        "ethics statement", "ethical considerations", "broader impact",
        "broader impacts", "impact statement", "ethics", "broader impact statement",
    ],
    "acknowledgments": ["acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement"],
    "author_contributions": ["author contributions", "contributions", "authors' contributions"],
    "references": ["references", "bibliography", "reference"],
    "appendix": ["appendix", "appendices", "supplementary material", "supplementary materials", "appendix a"],
}

ALIAS_TO_CATEGORY = {
    alias: category for category, aliases in CANONICAL_SECTIONS.items() for alias in aliases
}

LEADING_NUMBER_RE = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.)\s+")
TRAILING_PUNCT_RE = re.compile(r"[.,:;]$")


def normalize_heading(line):
    s = line.strip()
    s = LEADING_NUMBER_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def find_heading_candidates(lines):
    """Yield (line_index, normalized_text) for lines with a blank line
    (or file boundary) immediately before AND after, non-empty, and not
    ending in sentence/run-in punctuation."""
    n = len(lines)
    for i, line in enumerate(lines):
        text = line.strip()
        if not text or len(text) > 80:
            continue
        if TRAILING_PUNCT_RE.search(text):
            continue
        before_ok = i == 0 or not lines[i - 1].strip()
        after_ok = i == n - 1 or not lines[i + 1].strip()
        if before_ok and after_ok:
            yield i, normalize_heading(text)


def split_sections(text, abstract_min_chars):
    lines = text.split("\n")
    boundaries = []  # (line_idx, category, raw_heading)
    for i, norm in find_heading_candidates(lines):
        category = ALIAS_TO_CATEGORY.get(norm.lower())
        if category:
            boundaries.append((i, category, lines[i].strip()))

    def line_start_char(line_idx):
        return sum(len(l) + 1 for l in lines[:line_idx])

    sections = []
    if not boundaries:
        preamble_end = len(text)
    else:
        preamble_end = line_start_char(boundaries[0][0])
    preamble = text[:preamble_end]

    # Blank-line-delimited paragraph START offsets within preamble (not full
    # (start, end) spans with content excluded -- see below): used only to
    # decide where the abstract begins. The actual front/abstract split is
    # a single slice at that offset, so front + abstract == preamble exactly
    # (no character loss from separator whitespace re.split() would consume
    # and \n\n-rejoin wouldn't exactly reproduce).
    para_starts = [0]
    para_spans = []  # (start, end) excluding the separator, for length checks only
    prev = 0
    for m in re.finditer(r"\n\s*\n", preamble):
        para_spans.append((prev, m.start()))
        para_starts.append(m.end())
        prev = m.end()
    para_spans.append((prev, len(preamble)))

    abstract_idx = None
    for idx in range(len(para_spans) - 1, -1, -1):
        start, end = para_spans[idx]
        if end - start >= abstract_min_chars:
            abstract_idx = idx
            break
    # Always append front_matter/abstract, even if whitespace-only or empty
    # -- skipping an empty-looking piece would silently drop those
    # characters from chars_covered, breaking the exact-coverage guarantee.
    # Consumers can filter empty/whitespace-only text themselves.
    if abstract_idx is None:
        sections.append({"category": "front_matter", "heading": None, "text": preamble})
    else:
        split_at = para_starts[abstract_idx]
        front, abstract = preamble[:split_at], preamble[split_at:]
        sections.append({"category": "front_matter", "heading": None, "text": front})
        sections.append({"category": "abstract", "heading": None, "text": abstract})

    for bi, (line_idx, category, raw_heading) in enumerate(boundaries):
        start = line_start_char(line_idx)
        end = line_start_char(boundaries[bi + 1][0]) if bi + 1 < len(boundaries) else len(text)
        sections.append({"category": category, "heading": raw_heading, "text": text[start:end]})

    return sections


def process_file(path, abstract_min_chars):
    text = path.read_text(errors="replace")
    sections = split_sections(text, abstract_min_chars)
    return {
        "doc_id": path.stem,
        "sections": sections,
        "n_recognized_headings": sum(1 for s in sections if s["heading"] is not None),
        "total_chars": len(text),
        "chars_covered": sum(len(s["text"]) for s in sections),
    }


def report_unmatched(files, top_n):
    counts = Counter()
    for path in files:
        text = path.read_text(errors="replace")
        lines = text.split("\n")
        for _, norm in find_heading_candidates(lines):
            if norm.lower() not in ALIAS_TO_CATEGORY and norm:
                counts[norm] += 1
    print(f"top {top_n} candidate headings NOT in the whitelist (line-shaped like a heading, "
          f"blank before/after, no trailing punctuation, but no alias match):", file=sys.stderr)
    for text, n in counts.most_common(top_n):
        print(f"  {n:4d}  {text!r}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text_dir")
    p.add_argument("--glob", default="*.txt")
    p.add_argument("--jsonl", default=None)
    p.add_argument("--abstract-min-chars", type=int, default=300)
    p.add_argument("--report-unmatched", action="store_true",
                    help="print the most common unrecognized heading-shaped lines and exit")
    p.add_argument("--top", type=int, default=40)
    args = p.parse_args()

    text_dir = Path(args.text_dir)
    files = sorted(text_dir.glob(args.glob))
    if not files:
        sys.exit(f"no files matching {args.glob} under {text_dir}")

    if args.report_unmatched:
        report_unmatched(files, args.top)
        return

    if not args.jsonl:
        sys.exit("--jsonl is required unless --report-unmatched is given")

    n_missing_abstract = 0
    heading_hist = Counter()
    with open(args.jsonl, "w") as out:
        for f in files:
            record = process_file(f, args.abstract_min_chars)
            assert record["chars_covered"] == record["total_chars"], \
                f"{f.name}: section split lost/duplicated characters (bug, not a data issue)"
            if not any(s["category"] == "abstract" for s in record["sections"]):
                n_missing_abstract += 1
            for s in record["sections"]:
                heading_hist[s["category"]] += 1
            out.write(json.dumps(record) + "\n")

    print(f"DONE {len(files)} papers -> {args.jsonl}", file=sys.stderr)
    print(f"missing abstract (no paragraph >= {args.abstract_min_chars} chars in preamble): "
          f"{n_missing_abstract}/{len(files)}", file=sys.stderr)
    print("section category coverage (papers with >=1 section of that category):", file=sys.stderr)
    for cat, n in heading_hist.most_common():
        print(f"  {n:4d}  {cat}", file=sys.stderr)


if __name__ == "__main__":
    main()

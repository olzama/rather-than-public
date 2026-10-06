#!/usr/bin/env python3
r"""
Turn GROBID's TEI XML (from process_acl2019_grobid.py) into the same
per-paper section schema split_paper_sections.py produces for arXiv
2026, so both corpora can be consumed the same way.

Unlike split_paper_sections.py's text-line heuristics (needed because
plain detex'd text has no structural markup), GROBID's <div> elements
in <body> ARE real structural divisions -- a scholarly-PDF parser
trained for exactly this task, not a heuristic patched onto pdftotext's
column-scrambled output (see process_acl2019_grobid.py's docstring for
why the latter can't support section-splitting on this corpus at all).
So every <div> becomes a section here, whether or not its <head> text
matches the canonical whitelist -- an unmatched head still gets category
"other" with its raw heading preserved, rather than being silently
merged into the previous section as split_paper_sections.py does for an
unmatched candidate line (there, an unmatched line might not even BE a
real heading; here, the div boundary itself is already trustworthy).

Reuses ALIAS_TO_CATEGORY from split_paper_sections.py so the category
taxonomy (introduction/related_work/method/...) is identical across both
corpora -- deliberately imported, not copied, so they can't drift apart.

Usage:
    python3 parse_grobid_tei.py <tei_dir> --jsonl out.jsonl

Writes one JSON record per paper (same shape as split_paper_sections.py):
    {"doc_id": ..., "sections": [{"category": ..., "heading": <str or
     null>, "text": ...}, ...], "n_recognized_headings": ...,
     "total_chars": <None -- GROBID's XML has no single "original text
     length" to check coverage against; chars_covered sums the sections
     instead>, "chars_covered": ..., "n_references": <int>}

No coverage self-check against original text length here (unlike
split_paper_sections.py) -- GROBID's XML doesn't preserve a 1:1 mapping
back to raw PDF text (tables/figures/footnotes are handled separately,
not simply omitted from a linear stream), so "every character
accounted for" isn't a meaningful invariant to assert on this path.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_paper_sections import ALIAS_TO_CATEGORY, normalize_heading  # noqa: E402

TEI_NS = "{http://www.tei-c.org/ns/1.0}"


def local_tag(elem):
    return elem.tag.rsplit("}", 1)[-1]


def div_text(div):
    """All text inside a <div> except its own <head>, tags stripped,
    normalized whitespace."""
    parts = []
    for child in div:
        if local_tag(child) == "head":
            continue
        parts.append("".join(child.itertext()))
    text = " ".join(" ".join(p.split()) for p in parts if p.strip())
    return text


def div_heading(div):
    head = div.find(f"{TEI_NS}head")
    if head is None:
        return None
    return "".join(head.itertext()).strip()


def classify(heading):
    if heading is None:
        return "other"
    return ALIAS_TO_CATEGORY.get(normalize_heading(heading).lower(), "other")


def leaf_divs(parent, inherited_type=None):
    """Yield (div, type) for each <div> under parent that itself contains
    no nested <div> -- a "wrapper" div (no direct text, just grouping
    child divs, e.g. <div type="acknowledgement"><div><head>...) is
    recursed into rather than also yielded itself, which would otherwise
    double-count its children's text (found via D19-1004's acknowledgement
    div: the wrapper's own itertext() already includes its nested div's
    text, so processing both yields the same content twice)."""
    for div in parent.findall(f"{TEI_NS}div"):
        div_type = div.get("type") or inherited_type
        nested = div.findall(f"{TEI_NS}div")
        if nested:
            yield from leaf_divs(div, inherited_type=div_type)
        else:
            yield div, div_type


def parse_tei(path):
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        return None, f"XML_PARSE_ERROR: {e}"
    root = tree.getroot()

    sections = []

    abstract_el = root.find(f".//{TEI_NS}profileDesc/{TEI_NS}abstract")
    if abstract_el is not None:
        text = " ".join(" ".join(t.split()) for t in abstract_el.itertext() if t.strip())
        sections.append({"category": "abstract", "heading": None, "text": text})

    body = root.find(f".//{TEI_NS}text/{TEI_NS}body")
    if body is not None:
        for div, _ in leaf_divs(body):
            heading = div_heading(div)
            text = div_text(div)
            if not text.strip():
                continue
            sections.append({"category": classify(heading), "heading": heading, "text": text})

    n_references = 0
    back = root.find(f".//{TEI_NS}text/{TEI_NS}back")
    if back is not None:
        for div, div_type in leaf_divs(back):
            heading = div_heading(div)
            if div_type == "references":
                n_references = len(div.findall(f".//{TEI_NS}biblStruct"))
                continue  # bibliography entries aren't prose; count only
            text = div_text(div)
            if not text.strip():
                continue
            category = "acknowledgments" if div_type == "acknowledgement" else classify(heading)
            if category == "other" and div_type == "annex":
                category = "appendix"
            sections.append({"category": category, "heading": heading, "text": text})

    return {
        "doc_id": path.stem.replace(".tei", ""),
        "sections": sections,
        "n_recognized_headings": sum(1 for s in sections if s["category"] != "other"),
        "chars_covered": sum(len(s["text"]) for s in sections),
        "n_references": n_references,
    }, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tei_dir")
    p.add_argument("--jsonl", required=True)
    args = p.parse_args()

    tei_dir = Path(args.tei_dir)
    files = sorted(tei_dir.glob("*.tei.xml"))
    if not files:
        sys.exit(f"no *.tei.xml files under {tei_dir}")

    from collections import Counter
    heading_hist = Counter()
    n_errors = 0
    with open(args.jsonl, "w") as out:
        for f in files:
            record, error = parse_tei(f)
            if error:
                print(f"{f.name}: {error}", file=sys.stderr)
                n_errors += 1
                continue
            for s in record["sections"]:
                heading_hist[s["category"]] += 1
            out.write(json.dumps(record) + "\n")

    print(f"DONE {len(files) - n_errors} papers -> {args.jsonl} ({n_errors} XML parse errors)",
          file=sys.stderr)
    print("section category coverage (papers with >=1 section of that category):", file=sys.stderr)
    for cat, n in heading_hist.most_common():
        print(f"  {n:4d}  {cat}", file=sys.stderr)


if __name__ == "__main__":
    main()

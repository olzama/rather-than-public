#!/usr/bin/env python3
r"""
Body length (words, \w+ tokens) per corpus: the corpus-sizes table.

ACL 2019 papers are split by published page count (short <= 7 pages,
long >= 9; 8 pages or no page metadata left out of the split), from the
Anthology bibliography export (--bib, anthology+abstracts.bib.gz), falling
back to the Anthology XML; dataset (2) by body length (short 500-3,500 words, long > 3,500;
under 500 words left out of the split). The version-diff rows use v1 and
the latest version of each paper with both. --exclude drops documents
listed in language_filter.py's output.

Usage:
    python3 corpus_sizes.py ../data --anthology-xml ../data/acl2019/anthology_xml \
        [--bib anthology+abstracts.bib.gz] [--exclude ../data/excluded_documents.tsv]
"""
import argparse
import gzip
import re
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

WORD_RE = re.compile(r"\w+")
VERSION_RE = re.compile(r"^(.*)_v(\d+)$")
OLD_ID_RE = re.compile(r"^([A-Z])(\d\d)-(\d{4})$")
NEW_ID_RE = re.compile(r"^(\d{4}\.[a-z0-9]+)-(\w+)\.(\d+)$")


def page_counts(xml_dir):
    """doc_id -> number of published pages, from the Anthology per-collection XML."""
    pages = {}
    for f in Path(xml_dir).glob("*.xml"):
        coll = f.stem
        for vol in ET.parse(f).getroot().iter("volume"):
            vid = vol.get("id")
            for paper in vol.iter("paper"):
                p = paper.findtext("pages")
                if not p:
                    continue
                nums = re.findall(r"\d+", p)
                if len(nums) != 2:
                    continue
                n = int(nums[1]) - int(nums[0]) + 1
                pid = paper.get("id")
                if re.match(r"^\d{4}\.", coll):
                    doc = f"{coll}-{vid}.{pid}"
                elif coll.startswith("W"):
                    doc = f"{coll}-{int(vid):02d}{int(pid):02d}"
                else:
                    doc = f"{coll}-{vid}{int(pid):03d}"
                pages[doc] = n
    return pages


def bib_page_counts(path):
    """doc_id -> number of published pages, from the bibliography export's url and pages fields."""
    pages = {}
    text = gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
    for entry in text.split("\n@")[1:]:
        url = re.search(r'url\s*=\s*"https?://aclanthology\.org/([^/"]+)/?"', entry)
        pg = re.search(r'pages\s*=\s*"(\d+)-+(\d+)"', entry)
        if url and pg:
            pages[url.group(1)] = int(pg.group(2)) - int(pg.group(1)) + 1
    return pages


def words(path):
    return len(WORD_RE.findall(path.read_text(errors="replace")))


def row(label, vals):
    return f"{label:28s} {len(vals):6d} {statistics.mean(vals):8.0f} {statistics.median(vals):8.0f}" if vals else f"{label:28s} 0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--anthology-xml", required=True)
    ap.add_argument("--bib")
    ap.add_argument("--exclude")
    args = ap.parse_args()
    d = Path(args.data_dir)
    skip = defaultdict(set)
    if args.exclude:
        for line in list(open(args.exclude))[1:]:
            c, doc = line.split("\t")[:2]
            skip[c].add(doc)
    pages = page_counts(args.anthology_xml)
    if args.bib:
        pages.update(bib_page_counts(args.bib))

    acl = {f.stem: words(f) for f in (d / "acl2019" / "text_body").glob("*.txt") if f.stem not in skip["acl2019"]}
    arx = {f.stem: words(f) for f in (d / "arxiv2026" / "text_body").glob("*.txt") if f.stem not in skip["arxiv2026"]}
    versions = defaultdict(dict)
    for f in (d / "arxiv_versions" / "text_body").glob("*_v*.txt"):
        m = VERSION_RE.match(f.stem)
        if m and f.stem not in skip["arxiv_versions"]:
            versions[m.group(1)][int(m.group(2))] = f
    v1 = [words(vs[1]) for vs in versions.values() if 1 in vs]
    latest = [words(vs[max(vs)]) for vs in versions.values() if len(vs) >= 2 and max(vs) > 1]

    print(f"{'corpus':28s} {'N':>6s} {'mean':>8s} {'median':>8s}")
    print(row("ACL 2019 (1), all", list(acl.values())))
    print(row("  short (<=7pp)", [w for k, w in acl.items() if pages.get(k, 0) and pages[k] <= 7]))
    print(row("  long (>=9pp)", [w for k, w in acl.items() if pages.get(k, 0) >= 9]))
    print(f"  (no page metadata: {sum(k not in pages for k in acl)}; 8 pages: {sum(pages.get(k) == 8 for k in acl)})")
    print(row("arXiv 2026 (2), all", list(arx.values())))
    print(row("  short (500-3.5k)", [w for w in arx.values() if 500 <= w <= 3500]))
    print(row("  long (>3.5k)", [w for w in arx.values() if w > 3500]))
    print(row("version-diff v1", v1))
    print(row("version-diff latest", latest))


if __name__ == "__main__":
    main()

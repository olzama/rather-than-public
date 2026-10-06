#!/usr/bin/env python3
r"""
Add a "title" field to every record of the ACL 2019 and arXiv 2026
sections.jsonl.gz files, in place. Run after parse_grobid_tei.py /
split_paper_sections.py, which write records without titles.

Titles:
  - acl2019: the official ACL Anthology title, read from the Anthology's
    per-volume XML (github.com/acl-org/acl-anthology, data/xml/). The
    first line of the pdftotext output is not usable: titles often wrap
    onto a second line with no reliable end marker. XML files are cached
    in --anthology-cache so reruns don't refetch.
  - arxiv2026: the title field of harvest_arxiv2026.py's candidates TSV.

Usage:
    python3 add_titles_to_sections.py <data_dir> [--anthology-cache <dir>]

<data_dir> holds acl2019/ and arxiv2026/. Each record gets "title"
right after "doc_id" (null if no title is found); rerunning overwrites
it. Coverage counts go to stderr.
"""
import argparse
import csv
import gzip
import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ANTHOLOGY_XML_URL = "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/{}.xml"


def norm(text):
    return " ".join(text.split()) if text else None


def read_records(sections_path):
    with gzip.open(sections_path, "rt") as f:
        return [json.loads(line) for line in f]


def anthology_collection(doc_id):
    # "P19-1176" -> "P19"; "2019.ccnlg-1.1" -> "2019.ccnlg"
    return doc_id.split("-")[0]


def anthology_paper_id(collection, volume, paper):
    if collection[0].isdigit():
        return f"{collection}-{volume}.{paper}"
    if collection.startswith("W") or int(volume) >= 10:
        return f"{collection}-{int(volume):02d}{int(paper):02d}"
    return f"{collection}-{volume}{int(paper):03d}"


def fetch_collection_xml(collection, cache_dir):
    path = cache_dir / f"{collection}.xml"
    if not path.exists():
        with urllib.request.urlopen(ANTHOLOGY_XML_URL.format(collection), timeout=60) as resp:
            path.write_bytes(resp.read())
        time.sleep(1)
    return path


def load_anthology_titles(collections, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    titles = {}
    for collection in sorted(collections):
        root = ET.parse(fetch_collection_xml(collection, cache_dir)).getroot()
        for volume in root.findall("volume"):
            for paper in volume.findall("paper"):
                title_el = paper.find("title")
                if title_el is None:
                    continue
                pid = anthology_paper_id(collection, volume.get("id"), paper.get("id"))
                titles[pid] = norm("".join(title_el.itertext()))
    return titles


def load_arxiv_titles(candidates_path):
    with open(candidates_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return {row["arxiv_id"]: norm(row["title"]) for row in reader}


def write_titles(sections_path, records, titles):
    tmp_path = sections_path.with_suffix(".tmp")
    with gzip.open(tmp_path, "wt") as out:
        for record in records:
            rest = {k: v for k, v in record.items() if k not in ("doc_id", "title")}
            record = {"doc_id": record["doc_id"], "title": titles.get(record["doc_id"]), **rest}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(sections_path)
    missing = sorted(r["doc_id"] for r in records if r["doc_id"] not in titles)
    n = len(records)
    print(f"{sections_path}: title {n - len(missing)}/{n}", file=sys.stderr)
    if missing:
        print(f"  no title: {' '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--anthology-cache", default=None,
                   help="where to cache Anthology XML (default: <data_dir>/acl2019/anthology_xml)")
    args = p.parse_args()
    data_dir = Path(args.data_dir)

    acl_path = data_dir / "acl2019" / "sections.jsonl.gz"
    acl_records = read_records(acl_path)
    cache_dir = Path(args.anthology_cache) if args.anthology_cache else data_dir / "acl2019" / "anthology_xml"
    acl_titles = load_anthology_titles({anthology_collection(r["doc_id"]) for r in acl_records}, cache_dir)
    write_titles(acl_path, acl_records, acl_titles)

    arxiv_path = data_dir / "arxiv2026" / "sections.jsonl.gz"
    arxiv_records = read_records(arxiv_path)
    arxiv_titles = load_arxiv_titles(data_dir / "arxiv2026" / "arxiv2026_candidates.tsv")
    write_titles(arxiv_path, arxiv_records, arxiv_titles)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Harvest arXiv cs.CL submissions for 2026 and keep only those whose arXiv
`comment` field mentions an *ACL venue (ACL, EMNLP, NAACL, CoNLL, TACL,
Findings), as a cheap pre-filter before downloading LaTeX sources to check
for actual acl.sty usage.

Queried month-by-month: arXiv's search API pagination becomes unreliable
past ~10,000 results within a single query, and cs.CL/2026 as a whole
exceeds that, but no single month does.

Usage:
    python3 harvest_arxiv2026.py <out_candidates.tsv> [--until YYYYMMDD]
"""
import argparse
import calendar
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

VENUE_RE = re.compile(r"\b(ACL|EMNLP|NAACL|CoNLL|TACL|Findings)\b", re.I)

BASE = "https://export.arxiv.org/api/query"
PAGE_SIZE = 200


def month_ranges(until):
    until_year, until_month, until_day = int(until[:4]), int(until[4:6]), int(until[6:8])
    for month in range(1, until_month + 1):
        last_day = until_day if month == until_month else calendar.monthrange(until_year, month)[1]
        start = f"{until_year}{month:02d}01000000"
        end = f"{until_year}{month:02d}{last_day:02d}235959"
        yield start, end


def fetch_page(query, start, page_size):
    url = (
        f"{BASE}?search_query={query}&start={start}&max_results={page_size}"
        "&sortBy=submittedDate&sortOrder=ascending"
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except Exception as e:
            print(f"FETCH_ERROR start={start} attempt={attempt} {e}", file=sys.stderr)
            time.sleep(5)
    return None


def harvest(out_path, until):
    seen = 0
    matched = 0
    with open(out_path, "w") as out:
        out.write("arxiv_id\tversion\ttitle\tcomment\n")
        for month_start, month_end in month_ranges(until):
            query = f"cat:cs.CL+AND+submittedDate:[{month_start}+TO+{month_end}]"
            start = 0
            total = None
            while total is None or start < total:
                data = fetch_page(query, start, PAGE_SIZE)
                if data is None:
                    print(f"GIVING_UP month={month_start[:6]} start={start}", file=sys.stderr)
                    break

                root = ET.fromstring(data)
                if total is None:
                    tr = root.find("opensearch:totalResults", NS)
                    total = int(tr.text) if tr is not None else 0
                    print(f"month={month_start[:6]} total_results={total}", file=sys.stderr)

                entries = root.findall("atom:entry", NS)
                if not entries:
                    break
                for e in entries:
                    seen += 1
                    id_el = e.find("atom:id", NS)
                    title_el = e.find("atom:title", NS)
                    comment_el = e.find("arxiv:comment", NS)
                    arxiv_url = id_el.text.strip() if id_el is not None else ""
                    m = re.search(r"abs/([^v]+)v?(\d*)$", arxiv_url)
                    aid = m.group(1) if m else arxiv_url
                    ver = m.group(2) if m else ""
                    title = (title_el.text or "").strip().replace("\n", " ").replace("\t", " ")
                    comment = (
                        (comment_el.text or "").strip().replace("\n", " ").replace("\t", " ")
                        if comment_el is not None
                        else ""
                    )
                    if comment and VENUE_RE.search(comment):
                        matched += 1
                        out.write(f"{aid}\t{ver}\t{title}\t{comment}\n")
                        out.flush()
                start += PAGE_SIZE
                print(
                    f"progress month={month_start[:6]} seen={seen} matched={matched} "
                    f"month_start={start}/{total}",
                    file=sys.stderr,
                )
                time.sleep(3)

    print(f"DONE seen={seen} matched={matched}", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_path")
    p.add_argument("--until", default=time.strftime("%Y%m%d"), help="YYYYMMDD, default today")
    args = p.parse_args()
    harvest(args.out_path, args.until)

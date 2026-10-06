#!/usr/bin/env python3
"""
Extract ACL Anthology IDs for all 2019 entries from the anthology bulk
bibliography, excluding front matter (volume covers, tables of contents)
and EXCLUDED_IDS.

Front matter is identified by BibTeX entry type (@proceedings / @book,
vs. @inproceedings / @article for papers), not by title: front-matter
titles are the volume's booktitle, which is not always English
("Actes de la Conférence ...").

Usage:
    curl -s -o anthology.bib.gz https://aclanthology.org/anthology+abstracts.bib.gz
    gunzip -k anthology.bib.gz
    python3 extract_acl2019_ids.py anthology.bib > acl2019_ids.txt
"""
import re
import sys

# Anthology entries whose PDF is not the listed paper.
EXCLUDED_IDS = {
    "2019.iwslt-1.4",  # PDF is the ESPnet-ST IWSLT 2021 system paper
}

FRONT_MATTER_TYPES = {"proceedings", "book"}


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <anthology.bib>")

    with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
        content = f.read()

    entries = content.split("\n@")
    for i, e in enumerate(entries):
        if i > 0:
            e = "@" + e
        m_year = re.search(r'\byear\s*=\s*"?\{?(\d{4})\}?"?', e)
        if not (m_year and m_year.group(1) == "2019"):
            continue
        m_url = re.search(r'\burl\s*=\s*[{"](https://aclanthology\.org/([^/}"]+))/?[}"]', e)
        m_title = re.search(r'\btitle\s*=\s*[{"](.+?)[}"]\s*,?\s*\n', e, re.S)
        if not m_url:
            continue
        anthology_id = m_url.group(2)
        entry_type = re.match(r"@(\w+)", e)
        if (entry_type and entry_type.group(1).lower() in FRONT_MATTER_TYPES) or anthology_id in EXCLUDED_IDS:
            continue
        title = m_title.group(1).strip() if m_title else ""
        if re.match(r"^(proceedings of|.*workshop program|.*front matter)", title, re.I):
            continue
        print(anthology_id)


if __name__ == "__main__":
    main()

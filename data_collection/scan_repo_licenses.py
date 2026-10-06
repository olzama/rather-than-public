#!/usr/bin/env python3
r"""
Classify each cloned repo's license by looking for a root-level LICENSE
file and matching its text against known license boilerplate. This
exists because build_code_paper_dataset.py's output embeds repo
README/code text verbatim, and a repo with no LICENSE file grants no
redistribution rights under default copyright -- being public on GitHub
only permits viewing/forking there, not copying the source elsewhere.
See ../README.md's note on the license breakdown before treating any
downstream dataset as freely shareable.

Text matching, not GitHub's license API (github.com/<owner>/<repo> license
detection) -- deliberately, since this works offline against the repos
already cloned by download_code_release_repos.py and doesn't need a
token or hit GitHub's rate limits again.

Usage:
    python3 scan_repo_licenses.py <repos_dir> <out.json>

<repos_dir> is download_code_release_repos.py's <out_dir>/repos (dirs
named "<owner>__<repo>"). Writes a JSON object {"<owner>/<repo>":
"<license>"}; license is one of the recognized SPDX-ish labels below,
"unrecognized_license_text" (a LICENSE file exists but didn't match any
pattern), or "NO_LICENSE_FILE".
"""
import json
import re
import sys
from pathlib import Path

LICENSE_NAMES = {
    "license", "licence", "copying", "license.md", "licence.md",
    "license.txt", "licence.txt", "copying.md", "copying.txt", "unlicense",
}

PATTERNS = [
    ("MIT", re.compile(r"\bMIT License\b", re.I)),
    ("Apache-2.0", re.compile(r"Apache License[,\s]+Version 2\.0", re.I)),
    ("BSD-3-Clause", re.compile(r"Redistributions of source code must retain.*3\. Neither the name", re.S | re.I)),
    ("BSD-2-Clause", re.compile(r"Redistributions of source code must retain", re.I)),
    ("GPL-3.0", re.compile(r"GNU GENERAL PUBLIC LICENSE\s*\n*\s*Version 3", re.I)),
    ("GPL-2.0", re.compile(r"GNU GENERAL PUBLIC LICENSE\s*\n*\s*Version 2", re.I)),
    ("LGPL", re.compile(r"GNU LESSER GENERAL PUBLIC LICENSE", re.I)),
    ("AGPL", re.compile(r"GNU AFFERO GENERAL PUBLIC LICENSE", re.I)),
    ("MPL-2.0", re.compile(r"Mozilla Public License.*2\.0", re.I)),
    ("CC-BY-NC", re.compile(r"Attribution-NonCommercial", re.I)),
    ("CC-BY", re.compile(r"Creative Commons Attribution", re.I)),
    ("Unlicense", re.compile(r"This is free and unencumbered software", re.I)),
    ("ISC", re.compile(r"Permission to use, copy, modify, and(?:/or)? distribute this software", re.I)),
]


def classify(text):
    for name, rx in PATTERNS:
        if rx.search(text):
            return name
    return "unrecognized_license_text"


def find_license(repo_dir):
    for child in repo_dir.iterdir():
        if child.is_file() and child.name.lower() in LICENSE_NAMES:
            try:
                text = child.read_text(errors="replace")
            except OSError:
                continue
            return classify(text)
    return "NO_LICENSE_FILE"


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: scan_repo_licenses.py <repos_dir> <out.json>")
    repos_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    results = {}
    for repo_dir in sorted(repos_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        owner_repo = repo_dir.name.replace("__", "/", 1)
        results[owner_repo] = find_license(repo_dir)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    counts = {}
    for v in results.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"scanned {len(results)} repos -> {out_path}", file=sys.stderr)
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}", file=sys.stderr)


if __name__ == "__main__":
    main()

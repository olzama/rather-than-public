#!/usr/bin/env python3
"""
Search GitHub for public repositories that plausibly contain an ACL-style
LaTeX paper (i.e. an Overleaf project pushed to GitHub, or a paper written
directly with git), as candidates for mining phrase-level edit history
(see mine_edit_history.py and this directory's README).

We deliberately search for `filename:main.tex` combined with an ACL style
marker rather than just `acl.sty`/`acl_natbib.bst extension:*` on their own:
the broader queries are dominated (>90%) by AI-agent "paper writing skill"
template bundles that ship a copy of the ACL style files but are not
themselves papers. Restricting to `main.tex` cuts that noise dramatically
(68 vs. ~1000 results, in initial testing) at the cost of missing papers
that don't use that filename convention -- an acceptable trade for a
supplementary, hand-checked corpus rather than an exhaustive one.

Requires a GitHub token (code search is not available unauthenticated):
    export GITHUB_TOKEN=...   # a classic PAT with no special scopes needed

Usage:
    python3 search_github_repos.py <out_candidates.tsv> [--query Q]...

With no --query given, uses the default query described above. Pass
--query multiple times to run additional/alternate searches; results are
deduplicated by (repo, path).
"""
import argparse
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import json

API = "https://api.github.com/search/code"
DEFAULT_QUERIES = ['"\\usepackage{acl}" filename:main.tex']
PAGE_SIZE = 100
# GitHub code search: 10 requests/min for authenticated users.
SLEEP_BETWEEN_REQUESTS = 7


def search(query, token):
    token = token
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    seen = []
    page = 1
    while True:
        qs = urllib.parse.urlencode({"q": query, "per_page": PAGE_SIZE, "page": page})
        req = urllib.request.Request(f"{API}?{qs}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            print(f"HTTP_ERROR query={query!r} page={page}: {e.code} {e.read()[:300]}", file=sys.stderr)
            break
        items = data.get("items", [])
        if page == 1:
            print(f"query={query!r} total_count={data.get('total_count')}", file=sys.stderr)
        if not items:
            break
        for it in items:
            seen.append((it["repository"]["full_name"], it["path"]))
        if len(items) < PAGE_SIZE:
            break
        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_path")
    p.add_argument("--query", action="append", dest="queries", default=None)
    args = p.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN env var not set (GitHub code search requires authentication)")

    queries = args.queries or DEFAULT_QUERIES

    results = {}
    for i, q in enumerate(queries):
        for repo, path in search(q, token):
            results[(repo, path)] = None
        if i < len(queries) - 1:
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    with open(args.out_path, "w") as out:
        out.write("repo\tpath\n")
        for repo, path in sorted(results):
            out.write(f"{repo}\t{path}\n")

    print(f"DONE {len(results)} unique (repo, path) candidates -> {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

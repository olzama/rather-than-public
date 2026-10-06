#!/usr/bin/env python3
"""
For each (repo, path) candidate from search_github_repos.py, count commits
that touched that file, via the GitHub commits API. This is the signal we
actually care about: a repo containing ACL-style LaTeX is only useful for
mining phrase-level revisions if it was edited incrementally rather than
uploaded once.

Empirically (68 candidates from the default search_github_repos.py query,
checked 2026-09-11): 36/68 (53%) had exactly 1 commit touching the file
(a one-shot upload -- no edit signal at all), and only 10/68 had 5+
commits. So expect most candidates to be discarded at this step; that's
normal, not a bug.

Requires GITHUB_TOKEN in the environment (same token as the search step;
this uses the regular REST API, not code search, so the rate limit is the
standard 5000/hr).

Usage:
    python3 check_commit_history.py <in_candidates.tsv> <out_counts.tsv>
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com/repos/{repo}/commits"


def commit_count(repo, path, token):
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    qs = urllib.parse.urlencode({"path": path, "per_page": 100})
    url = f"{API.format(repo=repo)}?{qs}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"HTTP_ERROR repo={repo} path={path}: {e.code}", file=sys.stderr)
        return -1
    if not isinstance(data, list):
        return -1
    # Capped at 100 (one page); good enough to separate "1-shot upload"
    # from "actively edited" without needing to paginate further.
    return len(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("in_path")
    p.add_argument("out_path")
    args = p.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN env var not set")

    with open(args.in_path) as f:
        lines = [l.rstrip("\n").split("\t") for l in f.readlines()[1:] if l.strip()]

    with open(args.out_path, "w") as out:
        out.write("repo\tpath\tcommit_count\n")
        for repo, path in lines:
            n = commit_count(repo, path, token)
            out.write(f"{repo}\t{path}\t{n}\n")
            out.flush()
            time.sleep(0.3)

    print(f"DONE {len(lines)} candidates checked -> {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Clone the GitHub repos identified by ../counting/count_code_release_links.py
as a paper's own code release, and write a manifest linking each cloned
repo back to the paper(s) that cited it.

Does not require a GitHub token -- clones over plain https, --depth 1
(only the current snapshot; unlike version_history/mine_edit_history.py,
here we want the released code itself, not edit history). A repo cited
by more than one paper (rare, but possible if a doc_id/repo pair repeats
across corpora) is cloned once and gets one manifest row per citing
paper.

A matched_url of the form "github.com/<owner>" with no repo segment
(rare -- e.g. the extracted sentence trailed off before the repo name)
can't be cloned and is recorded with clone_status "no_repo_segment".

Usage:
    python3 download_code_release_repos.py <out_dir> <in.jsonl> [<in.jsonl> ...]

<in.jsonl> files are count_code_release_links.py --jsonl output (must
include a "corpus" field -- pass --corpus-name when generating them).
Writes cloned repos to <out_dir>/repos/<owner>__<repo>/ and one manifest
row per (corpus, doc_id) with an own-code link to <out_dir>/manifest.jsonl:

    {"corpus": ..., "doc_id": ..., "matched_url": ..., "owner_repo": ...,
     "local_dir": ..., "clone_status": "ok" | "failed" | "no_repo_segment",
     "error": "<git stderr, if failed>"}

Point <out_dir> outside this repo (e.g. /mnt/kesha/rather-than/data/) --
cloned repos don't belong in version control here, same reasoning as
version_history/README.md's note on <out_dir> for that pipeline.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def owner_repo(matched_url):
    """'github.com/<owner>/<repo>' -> '<owner>/<repo>', or None if the
    repo segment is missing."""
    parts = matched_url.split("/", 2)
    if len(parts) < 3 or not parts[2]:
        return None
    return f"{parts[1]}/{parts[2]}"


def clone(repo, repos_dir):
    org_name = repo.replace("/", "__")
    dest = repos_dir / org_name
    if dest.exists():
        return dest, "ok", None
    url = f"https://github.com/{repo}.git"
    print(f"cloning {repo}...", file=sys.stderr)
    try:
        result = subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", url, str(dest)],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"CLONE_TIMEOUT {repo}", file=sys.stderr)
        subprocess.run(["rm", "-rf", str(dest)])
        return None, "failed", "clone timed out after 120s"
    if result.returncode != 0:
        err = result.stderr.strip()[:300]
        print(f"CLONE_FAILED {repo}: {err}", file=sys.stderr)
        return None, "failed", err
    return dest, "ok", None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("in_jsonl", nargs="+")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    repos_dir = out_dir / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in args.in_jsonl:
        for line in open(path):
            r = json.loads(line)
            if r["own_code_link"]:
                records.append(r)

    # Written incrementally (one row per record, flushed immediately) so a
    # consumer can join against partial progress while this is still
    # running, and so an interrupted run doesn't lose already-cloned repos'
    # manifest rows.
    clone_cache = {}  # owner/repo -> (dest_or_None, status, error)
    manifest_path = out_dir / "manifest.jsonl"
    n_ok = n_failed = n_no_repo = 0
    unique_repos = set()
    with open(manifest_path, "w") as out:
        for r in records:
            repo = owner_repo(r["matched_url"])
            if repo is None:
                row = {
                    "corpus": r["corpus"], "doc_id": r["doc_id"],
                    "matched_url": r["matched_url"], "owner_repo": None,
                    "local_dir": None, "clone_status": "no_repo_segment", "error": None,
                }
                n_no_repo += 1
            else:
                if repo not in clone_cache:
                    clone_cache[repo] = clone(repo, repos_dir)
                dest, status, error = clone_cache[repo]
                row = {
                    "corpus": r["corpus"], "doc_id": r["doc_id"],
                    "matched_url": r["matched_url"], "owner_repo": repo,
                    "local_dir": str(dest) if dest else None,
                    "clone_status": status, "error": error,
                }
                unique_repos.add(repo)
                n_ok += status == "ok"
                n_failed += status == "failed"
            out.write(json.dumps(row) + "\n")
            out.flush()

    print(f"papers: {len(records)}  unique repos: {len(unique_repos)}  "
          f"cloned ok: {n_ok}  failed: {n_failed}  no_repo_segment: {n_no_repo}",
          file=sys.stderr)
    print(f"manifest -> {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

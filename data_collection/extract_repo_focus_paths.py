#!/usr/bin/env python3
r"""
Find papers that link to a specific directory/file inside their code repo
(github.com/<owner>/<repo>/tree|blob/<branch>/<path>) -- typical for
monorepos like AlibabaResearch/DAMO-ConvAI, where one paper's code is one
subdirectory among dozens. count_code_release_links.py reduces links to
owner/repo, so the path is recovered here from the paper text.

Only the URL is taken from the paper; build_code_summary_bundles.py
--focus-paths then ranks files under these paths first. No paper content
reaches the summary bundles.

Usage:
    python3 extract_repo_focus_paths.py <manifest.jsonl> <data_dir> <out.json>

<data_dir> holds "<corpus>/<text dir>/<doc_id>*" files; every text variant
directory listed in TEXT_DIRS is searched. Writes
{"owner/repo": ["subdir", ...]} for paths that exist in the local clone;
linked paths missing from the clone (moved/renamed since publication) are
reported on stderr and left out.
"""
import glob
import json
import os
import re
import sys

TEXT_DIRS = {
    "arxiv2026": ["text_noun_placeholder", "text_old_detex_bug"],
    "acl2019": ["text_old_layout_bug", "tei"],
}


def main():
    manifest, data_dir, out_path = sys.argv[1:4]
    focus = {}
    for line in open(manifest):
        row = json.loads(line)
        if row["clone_status"] != "ok":
            continue
        text = ""
        for d in TEXT_DIRS.get(row["corpus"], []):
            for p in glob.glob(os.path.join(data_dir, row["corpus"], d, row["doc_id"] + "*")):
                text += open(p, errors="replace").read()
        owner, repo = row["owner_repo"].split("/")
        link_re = re.compile(r"github\.com/" + re.escape(owner) + "/" + re.escape(repo)
                             + r"/(?:tree|blob)/[\w.\-]+/([\w./\-]+)", re.I)
        for m in link_re.finditer(text):
            sub = m.group(1).rstrip("./")
            if os.path.exists(os.path.join(row["local_dir"], sub)):
                paths = focus.setdefault(row["owner_repo"], [])
                if sub not in paths:
                    paths.append(sub)
            else:
                print(f"missing in clone: {row['owner_repo']}/{sub} ({row['doc_id']})", file=sys.stderr)
    json.dump(focus, open(out_path, "w"), indent=1, sort_keys=True)
    print(f"{len(focus)} repos with paper-linked focus paths -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

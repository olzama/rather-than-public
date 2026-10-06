#!/usr/bin/env python3
r"""
Join each paper's text with its own-code-release repo's README and source
files into one JSON record per paper, from the manifest written by
download_code_release_repos.py.

Repos are NOT embedded in full -- several of the ~700 cloned here are
multi-gigabyte (committed datasets/checkpoints alongside the code), so
"the code" here means: the root README, plus files under a whitelisted
set of code/config extensions, walked in sorted-path order and stopped
once --max-total-chars is hit (a repo's file order is otherwise
arbitrary, so truncation always drops from the same end rather than
randomly). Common non-code directories (data, checkpoints, node_modules,
.git, ...) are skipped outright rather than counted against the cap, so
the cap is spent on actual code, not on skipping noise. This is a
heuristic split, not a guarantee that every skipped file was
uninteresting or every included one was core logic.

Usage:
    python3 build_code_paper_dataset.py <manifest.jsonl> <data_dir> <out.jsonl> \
        [--licenses licenses.json] [--metadata-out metadata.jsonl] \
        [--max-file-chars 50000] [--max-total-chars 500000]

<data_dir> is the corpus root containing "<corpus>/text/<doc_id>.txt" for
each corpus named in the manifest (e.g. ../data, matching the "corpus"
field values "acl2019"/"arxiv2026" written by count_code_release_links.py
--corpus-name).

IMPORTANT -- license, before committing/sharing anywhere: see
scan_repo_licenses.py. As of 2026-09-15, ~48% of the cloned repos have no
LICENSE file at all, which grants no redistribution right under default
copyright even though the repo is public on GitHub. <out.jsonl> embeds
README/code text verbatim regardless of license -- treat it as a private
research artifact, not something to commit or share further, unless
you've filtered by license. --metadata-out writes a second file with
everything EXCEPT paper_text/readme/code_files (just doc_id/repo/license/
clone stats) -- pure facts, safe to commit regardless of license.

Only manifest rows with clone_status == "ok" are processed. Writes one
JSON record per paper to <out.jsonl>:

    {"corpus": ..., "doc_id": ..., "owner_repo": ..., "license": ...,
     "repo_url": ..., "paper_text": ..., "readme": <str or null>,
     "code_files": [{"path": "<repo-relative>", "content": ...}, ...],
     "n_files_total": <files matching the extension whitelist, after
                       directory exclusion>,
     "n_files_included": <files actually embedded before the cap>,
     "code_truncated": <bool -- hit --max-total-chars before n_files_total>,
     "total_code_chars": <sum of len(content) across code_files>}

"license" is "unknown" if --licenses wasn't passed.
"""
import argparse
import json
import sys
from pathlib import Path

EXCLUDED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".env",
    "data", "datasets", "dataset", "checkpoints", "checkpoint", "models",
    "weights", "logs", "log", "results", "outputs", "output",
    ".ipynb_checkpoints", "build", "dist", "assets", "images", "img",
    "figures", "wandb", "runs", "cache", ".cache", "saved_models",
    "pretrained", "ckpt", ".idea", ".vscode",
}

CODE_EXTENSIONS = {
    ".py", ".ipynb", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp",
    ".cc", ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".sh", ".bash",
    ".pl", ".r", ".jl", ".scala", ".lua", ".sql", ".yaml", ".yml",
    ".json", ".toml", ".cfg", ".ini", ".md", ".rst", ".proto", ".gradle",
    ".cmake", ".m", ".swift", ".kt",
}


def read_capped(path, max_chars):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read(max_chars + 1)[:max_chars]
    except OSError:
        return None


def find_readme(repo_dir, max_chars):
    for child in sorted(repo_dir.iterdir()):
        if child.is_file() and child.stem.lower() == "readme":
            return read_capped(child, max_chars)
    return None


def collect_code_files(repo_dir, max_file_chars, max_total_chars):
    candidates = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(repo_dir).parts[:-1]):
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if path.stem.lower() == "readme" and path.parent == repo_dir:
            continue  # already captured separately
        candidates.append(path)
    candidates.sort(key=lambda p: str(p.relative_to(repo_dir)))

    code_files = []
    total_chars = 0
    truncated = False
    for path in candidates:
        if total_chars >= max_total_chars:
            truncated = True
            break
        content = read_capped(path, max_file_chars)
        if content is None:
            continue
        code_files.append({"path": str(path.relative_to(repo_dir)), "content": content})
        total_chars += len(content)
    return code_files, len(candidates), truncated, total_chars


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("data_dir")
    p.add_argument("out_path")
    p.add_argument("--licenses", default=None,
                    help="scan_repo_licenses.py --out JSON ('owner/repo' -> license label); "
                         "adds a 'license' field to each record if given")
    p.add_argument("--metadata-out", default=None,
                    help="also write a second JSONL with everything except paper_text/readme/"
                         "code_files -- the safe-to-commit half when licenses are mixed/missing")
    p.add_argument("--max-file-chars", type=int, default=50_000)
    p.add_argument("--max-total-chars", type=int, default=500_000)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    licenses = json.load(open(args.licenses)) if args.licenses else {}
    n_written = n_missing_text = n_missing_repo = 0
    metadata_out = open(args.metadata_out, "w") if args.metadata_out else None
    with open(args.out_path, "w") as out:
        for line in open(args.manifest):
            row = json.loads(line)
            if row["clone_status"] != "ok":
                continue
            repo_dir = Path(row["local_dir"])
            if not repo_dir.is_dir():
                n_missing_repo += 1
                continue
            text_path = data_dir / row["corpus"] / "text" / f"{row['doc_id']}.txt"
            if not text_path.is_file():
                n_missing_text += 1
                continue
            paper_text = text_path.read_text(errors="replace")
            readme = find_readme(repo_dir, args.max_file_chars)
            code_files, n_total, truncated, total_chars = collect_code_files(
                repo_dir, args.max_file_chars, args.max_total_chars
            )
            record = {
                "corpus": row["corpus"],
                "doc_id": row["doc_id"],
                "owner_repo": row["owner_repo"],
                "license": licenses.get(row["owner_repo"], "unknown"),
                "repo_url": f"https://github.com/{row['owner_repo']}",
                "paper_text": paper_text,
                "readme": readme,
                "code_files": code_files,
                "n_files_total": n_total,
                "n_files_included": len(code_files),
                "code_truncated": truncated,
                "total_code_chars": total_chars,
            }
            out.write(json.dumps(record) + "\n")
            if metadata_out:
                meta = {k: v for k, v in record.items()
                        if k not in ("paper_text", "readme", "code_files")}
                metadata_out.write(json.dumps(meta) + "\n")
            n_written += 1

    if metadata_out:
        metadata_out.close()

    print(f"DONE {n_written} records -> {args.out_path} "
          f"(skipped {n_missing_repo} missing repo dirs, {n_missing_text} missing paper text)",
          file=sys.stderr)
    if args.metadata_out:
        print(f"metadata-only copy -> {args.metadata_out}", file=sys.stderr)


if __name__ == "__main__":
    main()

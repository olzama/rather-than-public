#!/usr/bin/env python3
r"""
Robust loader/CLI for build_code_paper_dataset.py's output (paper text +
README + code, one record per paper) -- the full, license-mixed dataset
that stays outside this repo (see ../README.md Section 5 and
../data/code_release_repos/README.md). This tool DOES live in the repo:
it's code, not the repos' copyrighted content, so every coauthor who has
this checkout and a copy of the dataset file (shared separately -- ask
whoever built it) can use it the same way, without re-deriving the
schema or writing their own JSONL-scanning loop.

Problems this solves over "just open the JSONL and loop":
  - The file is large (hundreds of MB); scanning it start-to-end for one
    doc_id is slow and easy to forget to do correctly (both an off-by-one
    line issue and just remembering the field names). A sidecar
    "<dataset>.index.json" (doc_id -> byte offset) makes single-paper
    lookups O(1) after the first build.
  - The index is validated against the dataset file's size+mtime on
    every load and rebuilt automatically if either changed, so a stale
    index left over from an earlier copy of the dataset can't silently
    return wrong offsets.
  - A malformed line (shouldn't happen, but this is shared-with-humans
    data) is reported with its line number and skipped, not a crash
    partway through indexing everyone else's papers.

Usage (all subcommands take the dataset path first):

    python3 code_paper_dataset.py index <dataset.jsonl>
        Build/refresh <dataset.jsonl>.index.json. Other subcommands do
        this automatically if the index is missing/stale, so you don't
        need to run this yourself first -- it's here for explicitly
        warming the index (e.g. right after copying the file) or CI.

    python3 code_paper_dataset.py get <dataset.jsonl> <doc_id> [--field NAME] [--full]
        Look up one paper. Default prints a summary (corpus, owner_repo,
        license, readme length, code file paths). --field paper_text (or
        readme, or any top-level key) prints just that field's raw text.
        --full prints the entire record as JSON.

    python3 code_paper_dataset.py list <dataset.jsonl> [--corpus C] [--license L]
        One "doc_id  corpus  owner_repo  license" line per matching
        record, for browsing/grepping.

    python3 code_paper_dataset.py stats <dataset.jsonl>
        Record counts by corpus and by license.

As a library:

    from code_paper_dataset import Dataset
    ds = Dataset("paper_code_dataset.jsonl")   # builds/loads the index
    record = ds.get("D19-1004")                # -> dict, or None
    for record in ds.iter(corpus="acl2019"):   # streams, doesn't load all
        ...
"""
import argparse
import json
import sys
from pathlib import Path


class DatasetError(Exception):
    pass


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise DatasetError(f"no such file: {self.path}")
        self.index_path = self.path.with_suffix(self.path.suffix + ".index.json")
        self._index = self._load_or_build_index()

    def _file_fingerprint(self):
        st = self.path.stat()
        return {"size": st.st_size, "mtime": st.st_mtime}

    def _load_or_build_index(self):
        fp = self._file_fingerprint()
        if self.index_path.is_file():
            try:
                cached = json.loads(self.index_path.read_text())
            except (OSError, json.JSONDecodeError):
                cached = None
            if cached and cached.get("fingerprint") == fp:
                return cached["doc_id_to_offset"]
            print(f"index stale or unreadable, rebuilding: {self.index_path}", file=sys.stderr)
        return self._build_index(fp)

    def _build_index(self, fp):
        index = {}
        n_bad_lines = 0
        n_duplicate = 0
        with open(self.path, "rb") as f:
            offset = f.tell()
            for lineno, raw in enumerate(f, start=1):
                if raw.strip():
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"skipping malformed JSON at line {lineno}", file=sys.stderr)
                        n_bad_lines += 1
                    else:
                        doc_id = row.get("doc_id")
                        if doc_id is None:
                            print(f"skipping line {lineno}: no 'doc_id' field", file=sys.stderr)
                            n_bad_lines += 1
                        elif doc_id in index:
                            print(f"duplicate doc_id {doc_id!r} at line {lineno} "
                                  f"(keeping the first)", file=sys.stderr)
                            n_duplicate += 1
                        else:
                            index[doc_id] = offset
                offset = f.tell()
        self.index_path.write_text(json.dumps(
            {"fingerprint": fp, "doc_id_to_offset": index}, indent=None
        ))
        print(f"indexed {len(index)} records -> {self.index_path} "
              f"({n_bad_lines} bad lines, {n_duplicate} duplicate doc_ids skipped)",
              file=sys.stderr)
        return index

    def __len__(self):
        return len(self._index)

    def doc_ids(self):
        return list(self._index.keys())

    def get(self, doc_id):
        offset = self._index.get(doc_id)
        if offset is None:
            return None
        with open(self.path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        return json.loads(line)

    def iter(self, corpus=None, license=None):
        """Stream every record in file order (doesn't require the index),
        optionally filtered. Use this instead of get() in a loop over
        many/most records -- one seek+read per record is wasted work
        when you're going to touch nearly all of them anyway."""
        with open(self.path, "r", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"skipping malformed JSON at line {lineno}", file=sys.stderr)
                    continue
                if corpus is not None and row.get("corpus") != corpus:
                    continue
                if license is not None and row.get("license") != license:
                    continue
                yield row


def cmd_index(args):
    ds = Dataset(args.dataset)
    print(f"{len(ds)} records indexed.", file=sys.stderr)


def cmd_get(args):
    ds = Dataset(args.dataset)
    record = ds.get(args.doc_id)
    if record is None:
        sys.exit(f"no record with doc_id {args.doc_id!r} ({len(ds)} doc_ids indexed -- "
                 f"try `list` to browse, or check corpus/doc_id spelling)")
    if args.full:
        print(json.dumps(record, indent=2))
    elif args.field:
        if args.field not in record:
            sys.exit(f"no field {args.field!r}; available: {sorted(record.keys())}")
        value = record[args.field]
        print(value if isinstance(value, str) else json.dumps(value, indent=2))
    else:
        print(f"doc_id:          {record['doc_id']}")
        print(f"corpus:          {record['corpus']}")
        print(f"owner_repo:      {record['owner_repo']}")
        print(f"license:         {record['license']}")
        print(f"repo_url:        {record['repo_url']}")
        print(f"paper_text:      {len(record['paper_text'])} chars")
        print(f"readme:          {len(record['readme']) if record['readme'] else 0} chars")
        print(f"code files:      {record['n_files_included']}/{record['n_files_total']}"
              f"{' (truncated)' if record['code_truncated'] else ''}")
        for cf in record["code_files"]:
            print(f"  - {cf['path']} ({len(cf['content'])} chars)")


def cmd_list(args):
    ds = Dataset(args.dataset)
    for row in ds.iter(corpus=args.corpus, license=args.license):
        print(f"{row['doc_id']}\t{row['corpus']}\t{row['owner_repo']}\t{row['license']}")


def cmd_stats(args):
    ds = Dataset(args.dataset)
    from collections import Counter
    by_corpus = Counter()
    by_license = Counter()
    for row in ds.iter():
        by_corpus[row["corpus"]] += 1
        by_license[row["license"]] += 1
    print("by corpus:")
    for k, v in by_corpus.most_common():
        print(f"  {v:4d}  {k}")
    print("by license:")
    for k, v in by_license.most_common():
        print(f"  {v:4d}  {k}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build/refresh the doc_id index")
    p_index.add_argument("dataset")
    p_index.set_defaults(func=cmd_index)

    p_get = sub.add_parser("get", help="look up one paper by doc_id")
    p_get.add_argument("dataset")
    p_get.add_argument("doc_id")
    p_get.add_argument("--field", default=None, help="print just this field (e.g. paper_text, readme)")
    p_get.add_argument("--full", action="store_true", help="print the entire record as JSON")
    p_get.set_defaults(func=cmd_get)

    p_list = sub.add_parser("list", help="list records, optionally filtered")
    p_list.add_argument("dataset")
    p_list.add_argument("--corpus", default=None)
    p_list.add_argument("--license", default=None)
    p_list.set_defaults(func=cmd_list)

    p_stats = sub.add_parser("stats", help="record counts by corpus/license")
    p_stats.add_argument("dataset")
    p_stats.set_defaults(func=cmd_stats)

    args = p.parse_args()
    try:
        args.func(args)
    except DatasetError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()

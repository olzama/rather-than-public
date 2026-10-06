#!/usr/bin/env python3
r"""
Re-run detex over already-downloaded arXiv LaTeX sources to replace
existing text/*.txt files, without re-fetching anything from arXiv.

Plain `detex` silently deletes math-mode content ($...$, \[...\]) and
\ref/\cite commands instead of substituting anything, so sentences that
quote or reference a symbol/table/citation come out with grammatically-
broken gaps (e.g. "Conditioning on  ensures that  and  characterize...").
Rather than have detex substitute its own placeholder (its -r flag hard-
codes "noun"/"noun verbs noun", confusing here specifically: these are
NLP papers, where "noun" is itself a common content word), this
pre-redacts \ref/\cite to a plain "[redacted]" placeholder in the .tex
source BEFORE running plain detex on it, then redacts math the same way
by collapsing detex's own between-word whitespace gaps -- the reliable
signature of a deleted math span -- into "[redacted]" (see
latex_redact.py for why math isn't hand-parsed at the source level).

For each existing <out_dir>/text/<name>.txt this locates the matching
raw source directory (checked first in <out_dir>/raw, then in
--reuse-dir/raw, matching how build_arxiv2026_corpus.py /
download_arxiv_versions.py originally found it), redacts a copy of its
source tree, runs plain detex on the redacted main .tex file, redacts
the math gaps in its output, and overwrites the .txt file if that
succeeds. <name> is either a bare arxiv id (dataset-(2)/arxiv2026
layout) or <arxiv_id>_v<N> (arxiv_versions layout, version parsed
straight from the filename).

Usage:
    python3 reextract_arxiv_detex.py <out_dir> [--reuse-dir <dir>]
"""
import argparse
import re
import sys
import tempfile
from pathlib import Path

from latex_redact import redact_math_gaps, redact_source_tree, run_detex

NAME_WITH_VERSION_RE = re.compile(r"^(.+)_v(\d+)$")


def find_main_tex(src_dir):
    tex_files = list(src_dir.rglob("*.tex"))
    if not tex_files:
        return None
    for f in tex_files:
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        if "\\documentclass" in text and "\\begin{document}" in text:
            return f
    return max(tex_files, key=lambda f: f.stat().st_size)


def find_src_dir(name, own_raw_dir, reuse_raw_dir):
    m = NAME_WITH_VERSION_RE.match(name)
    if m:
        aid, version = m.group(1), m.group(2)
        candidates = [f"{aid}_v{version}_src"]
    else:
        aid = name
        candidates = None  # figure out by globbing below

    for base in (own_raw_dir, reuse_raw_dir):
        if base is None:
            continue
        if candidates:
            for c in candidates:
                d = base / c
                if d.is_dir():
                    return d
        else:
            matches = sorted(base.glob(f"{aid}_v*_src"))
            if matches:
                # highest version number, if more than one is present
                matches.sort(key=lambda p: int(re.search(r"_v(\d+)_src$", p.name).group(1)))
                return matches[-1]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--reuse-dir", default=None, help="another pipeline's output dir to also check for raw/ (e.g. arxiv_versions for dataset-(2))")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    own_raw_dir = out_dir / "raw"
    text_dir = out_dir / "text"
    reuse_raw_dir = Path(args.reuse_dir) / "raw" if args.reuse_dir else None

    txt_files = sorted(text_dir.glob("*.txt"))
    print(f"{len(txt_files)} existing text files in {text_dir}", file=sys.stderr)

    no_src = detex_fail = changed = unchanged = 0
    for i, txt_path in enumerate(txt_files, 1):
        name = txt_path.stem
        src_dir = find_src_dir(name, own_raw_dir, reuse_raw_dir)
        if src_dir is None:
            no_src += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            redacted_dir = redact_source_tree(src_dir, Path(tmp) / "redacted")
            main_tex = find_main_tex(redacted_dir)
            if main_tex is None:
                no_src += 1
                continue
            stdout, ok, had_input_failure = run_detex(main_tex)
        if not ok or not stdout.strip():
            detex_fail += 1
            continue
        if had_input_failure:
            print(f"INPUT_FAILURE {name} (some \\input/\\include target(s) couldn't be "
                  f"opened; extracted text may be incomplete)", file=sys.stderr)
        new_text = redact_math_gaps(stdout)
        old = txt_path.read_text(errors="replace")
        if new_text != old:
            txt_path.write_text(new_text)
            changed += 1
        else:
            unchanged += 1
        if i % 200 == 0:
            print(f"progress: {i}/{len(txt_files)} changed={changed} unchanged={unchanged} "
                  f"no_src={no_src} detex_fail={detex_fail}", file=sys.stderr)

    print(f"DONE changed={changed} unchanged={unchanged} no_src={no_src} detex_fail={detex_fail} "
          f"(of {len(txt_files)} total)", file=sys.stderr)


if __name__ == "__main__":
    main()

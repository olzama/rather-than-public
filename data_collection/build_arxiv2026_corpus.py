#!/usr/bin/env python3
r"""
Build the actual dataset-(2) text corpus described in the paper's Corpora
section: for every candidate from harvest_arxiv2026.py (all ~1,985 papers
whose arXiv comment mentions an *ACL venue -- not just the multi-version
subset arxiv_versions/ works with), download the latest arXiv source,
confirm it actually uses the ACL style (acl.sty / \usepackage{acl}) rather
than just mentioning a venue in passing, and extract plain text via detex.

The comment-field filter is a cheap heuristic (see harvest_arxiv2026.py);
this is the step that turns "mentions ACL" into "confirmed ACL-templated,"
which is what dataset (2) is actually supposed to be.

Reuses sources arxiv_versions/download_arxiv_versions.py already
downloaded and extracted (for the multi-version subset it processed) so
those ~800 papers aren't fetched twice -- pass --reuse-dir pointing at
that pipeline's output directory.

Usage:
    python3 build_arxiv2026_corpus.py <candidates.tsv> <out_dir> \
        [--reuse-dir /mnt/kesha/rather-than/data/arxiv_versions]

<candidates.tsv> is harvest_arxiv2026.py's output (arxiv_id, version,
title, comment); `version` is the latest version number seen at harvest
time. Writes confirmed papers' text to <out_dir>/text/<arxiv_id>.txt, and
a summary log to <out_dir>/build.log.
"""
import argparse
import re
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from latex_redact import redact_math_gaps, redact_source_tree, run_detex

SLEEP_BETWEEN_REQUESTS = 3
ACL_STY_RE = re.compile(r"\\usepackage(\[[^\]]*\])?\{acl\}")


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


def get_source_dir(aid, version, own_raw_dir, reuse_raw_dir, log):
    """Return (src_dir, made_network_request). Reuses an already-extracted
    source directory if one exists (from a previous run of this script, or
    from arxiv_versions/'s own downloads), otherwise downloads + extracts."""
    name = f"{aid}_v{version}_src"
    if reuse_raw_dir is not None:
        candidate = reuse_raw_dir / name
        if candidate.is_dir():
            return candidate, False
    own_dir = own_raw_dir / name
    if own_dir.is_dir():
        return own_dir, False

    tarball = own_raw_dir / f"{aid}_v{version}.tar.gz"
    url = f"https://arxiv.org/e-print/{aid}v{version}"
    try:
        urllib.request.urlretrieve(url, tarball)
    except Exception as e:
        log.write(f"DOWNLOAD_FAIL {aid}v{version} {e}\n")
        return None, True

    own_dir.mkdir(exist_ok=True)
    try:
        with tarfile.open(tarball) as tf:
            tf.extractall(own_dir)
    except tarfile.ReadError:
        import gzip
        try:
            with gzip.open(tarball) as gz:
                (own_dir / "single.tex").write_bytes(gz.read())
        except Exception as e:
            log.write(f"EXTRACT_FAIL {aid}v{version} {e}\n")
            return None, True
    return own_dir, True


def uses_acl_style(src_dir):
    if list(src_dir.glob("acl.sty")) or list(src_dir.rglob("acl.sty")):
        return True
    for f in src_dir.rglob("*.tex"):
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        if ACL_STY_RE.search(text):
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidates_path")
    p.add_argument("out_dir")
    p.add_argument("--reuse-dir", default=None, help="arxiv_versions/ output dir to reuse downloads from")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    own_raw_dir = out_dir / "raw"
    text_dir = out_dir / "text"
    own_raw_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    reuse_raw_dir = Path(args.reuse_dir) / "raw" if args.reuse_dir else None

    with open(args.candidates_path) as f:
        rows = [l.rstrip("\n").split("\t") for l in f.readlines()[1:] if l.strip()]
    candidates = [(aid, int(ver)) for aid, ver, *_ in rows if ver.isdigit()]
    print(f"{len(candidates)} candidates", file=sys.stderr)

    log_path = out_dir / "build.log"
    downloaded = not_acl_style = extract_failed = confirmed = 0
    with open(log_path, "a") as log:
        for i, (aid, version) in enumerate(candidates, 1):
            out_txt = text_dir / f"{aid}.txt"
            if out_txt.exists():
                confirmed += 1
                continue

            src_dir, made_request = get_source_dir(aid, version, own_raw_dir, reuse_raw_dir, log)
            if made_request:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
            if src_dir is None:
                continue
            downloaded += 1

            if not uses_acl_style(src_dir):
                not_acl_style += 1
                log.write(f"NOT_ACL_STYLE {aid}v{version}\n")
                continue

            # Redact math/\ref/\cite to a plain "[redacted]" placeholder
            # BEFORE detex, rather than letting detex delete them (silent
            # gap) or substitute its own "noun" (confusing here: these are
            # NLP papers, where "noun" is itself a common content word --
            # checked 2026-09-14, see latex_redact.py).
            with tempfile.TemporaryDirectory() as tmp:
                work_dir = redact_source_tree(src_dir, Path(tmp) / "redacted")
                main_tex = find_main_tex(work_dir)
                if main_tex is None:
                    extract_failed += 1
                    log.write(f"NO_TEX_FOUND {aid}v{version}\n")
                    continue
                stdout, ok, had_input_failure = run_detex(main_tex)
            if not ok:
                extract_failed += 1
                log.write(f"DETEX_FAIL {aid}v{version}\n")
                continue
            if had_input_failure:
                log.write(f"INPUT_FAILURE {aid}v{version} (some \\input/\\include target(s) "
                          f"couldn't be opened; extracted text may be incomplete)\n")
            out_txt.write_text(redact_math_gaps(stdout))
            confirmed += 1

            if i % 50 == 0:
                log.write(
                    f"progress: {i}/{len(candidates)} downloaded={downloaded} "
                    f"not_acl_style={not_acl_style} extract_failed={extract_failed} "
                    f"confirmed={confirmed}\n"
                )
                log.flush()

    with open(log_path, "a") as log:
        log.write(
            f"DONE total={len(candidates)} downloaded={downloaded} "
            f"not_acl_style={not_acl_style} extract_failed={extract_failed} "
            f"confirmed={confirmed}\n"
        )
    print(f"DONE confirmed={confirmed}/{len(candidates)} -> {text_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

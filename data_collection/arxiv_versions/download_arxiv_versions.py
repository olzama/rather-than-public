#!/usr/bin/env python3
"""
For arXiv papers with more than one posted version, download the source
of the first and latest versions and extract plain text from each. This
is a much higher-yield alternative to mining public GitHub repos for
paper edit history (see ../version_history/): among the candidates
harvested by ../harvest_arxiv2026.py, ~41% already have 2+ arXiv
versions, vs. ~15% of GitHub candidates having any real commit history
at all. A version bump commonly corresponds to a preprint -> camera-ready
revision, which is exactly the kind of real, author-made edit we want to
diff for phrase-level changes (see diff_versions.py).

Does not require a GitHub token; arXiv e-prints are fetched anonymously.
Please keep the default pacing (one request every ~3s) to stay within
arXiv's requested rate limit.

Usage:
    python3 download_arxiv_versions.py <candidates.tsv> <out_dir> [--min-version 2]

<candidates.tsv> is the output of ../harvest_arxiv2026.py (columns:
arxiv_id, version, title, comment), where `version` is the latest
version number seen at harvest time. Writes text files to
<out_dir>/text/<id>_v1.txt and <out_dir>/text/<id>_v<version>.txt, and a
log to <out_dir>/download.log.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latex_redact import redact_math_gaps, redact_source_tree, run_detex

SLEEP_BETWEEN_REQUESTS = 3


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
    # fall back to the largest .tex file (likely the main document)
    return max(tex_files, key=lambda f: f.stat().st_size)


def download_and_extract(arxiv_id, version, work_dir, text_dir, log):
    """Returns (success, made_network_request) -- callers should only rate-limit-sleep
    when made_network_request is True, or a resumed run re-sleeps through every
    already-cached candidate before reaching new work (as a full-length run would)."""
    tarball = work_dir / f"{arxiv_id}_v{version}.tar.gz"
    src_dir = work_dir / f"{arxiv_id}_v{version}_src"
    out_txt = text_dir / f"{arxiv_id}_v{version}.txt"
    if out_txt.exists():
        return True, False

    url = f"https://arxiv.org/e-print/{arxiv_id}v{version}"
    try:
        urllib.request.urlretrieve(url, tarball)
    except urllib.error.HTTPError as e:
        log.write(f"DOWNLOAD_FAIL {arxiv_id}v{version} {e.code}\n")
        return False, True
    except Exception as e:
        log.write(f"DOWNLOAD_FAIL {arxiv_id}v{version} {e}\n")
        return False, True

    src_dir.mkdir(exist_ok=True)
    try:
        with tarfile.open(tarball) as tf:
            tf.extractall(src_dir)
    except tarfile.ReadError:
        # Some arXiv e-prints are a single gzipped .tex file, not a tar.
        import gzip
        try:
            with gzip.open(tarball) as gz:
                (src_dir / "single.tex").write_bytes(gz.read())
        except Exception as e:
            log.write(f"EXTRACT_FAIL {arxiv_id}v{version} {e}\n")
            return False, True

    # Redact math/\ref/\cite to a plain "[redacted]" placeholder BEFORE
    # detex, rather than letting detex delete them (silent gap, e.g.
    # "Conditioning on  ensures that  and  characterize...") or substitute
    # its own "noun" -- confusing here specifically, since these are NLP
    # papers where "noun" is itself a common content word. Checked
    # 2026-09-14, see latex_redact.py.
    with tempfile.TemporaryDirectory() as tmp:
        redacted_dir = redact_source_tree(src_dir, Path(tmp) / "redacted")
        main_tex = find_main_tex(redacted_dir)
        if main_tex is None:
            log.write(f"NO_TEX_FOUND {arxiv_id}v{version}\n")
            return False, True
        stdout, ok, had_input_failure = run_detex(main_tex)
    if not ok:
        log.write(f"DETEX_FAIL {arxiv_id}v{version}\n")
        return False, True
    if had_input_failure:
        log.write(f"INPUT_FAILURE {arxiv_id}v{version} (some \\input/\\include target(s) "
                  f"couldn't be opened; extracted text may be incomplete)\n")
    out_txt.write_text(redact_math_gaps(stdout))
    return True, True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidates_path")
    p.add_argument("out_dir")
    p.add_argument("--min-version", type=int, default=2)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    work_dir = out_dir / "raw"
    text_dir = out_dir / "text"
    work_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    with open(args.candidates_path) as f:
        rows = [l.rstrip("\n").split("\t") for l in f.readlines()[1:] if l.strip()]

    candidates = [(aid, int(ver)) for aid, ver, *_ in rows if ver.isdigit() and int(ver) >= args.min_version]
    print(f"{len(candidates)}/{len(rows)} candidates have >= {args.min_version} versions", file=sys.stderr)

    log_path = out_dir / "download.log"
    ok = fail = 0
    with open(log_path, "a") as log:
        for aid, latest_version in candidates:
            got_v1, req_v1 = download_and_extract(aid, 1, work_dir, text_dir, log)
            if req_v1:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
            got_latest, req_latest = download_and_extract(aid, latest_version, work_dir, text_dir, log)
            if req_latest:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
            if got_v1 and got_latest:
                ok += 1
            else:
                fail += 1
            if (ok + fail) % 20 == 0:
                log.write(f"progress: {ok + fail}/{len(candidates)} ok={ok} fail={fail}\n")
                log.flush()

    print(f"DONE ok={ok} fail={fail} -> {text_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

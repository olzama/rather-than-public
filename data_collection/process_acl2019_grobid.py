#!/usr/bin/env python3
r"""
Batch-process ACL 2019 PDFs through a running GROBID service to get TEI
XML with real document structure (title/author/abstract/body divs with
head text, in correct reading order) -- the fix for section-splitting
on this corpus, since the existing plain-text extraction
(download_acl2019.sh's `pdftotext -layout`-free output) can't support it:
checked by hand across several papers and found pdftotext's column
handling both decouples a heading's number from its title by dozens of
lines AND scrambles paragraph reading order itself (one paper's
subsection "3.2" appeared before "3.1" in the extracted text). GROBID is
a real scholarly-PDF structure parser (CRF + ML models trained for this
exact task), not a heuristic patch on pdftotext's output -- see
parse_grobid_tei.py for turning its TEI XML into the same section schema
split_paper_sections.py produces for arXiv 2026.

Requires a running GROBID instance (see tools/grobid/ -- built from
source with JDK 21 via `./gradlew run`, no Docker needed; default port
8070). Does NOT touch the existing acl2019/text/*.txt plain-text corpus
-- that one is unaffected by this issue for pure phrase/pattern COUNTING
(order doesn't matter for a count), only for section-level splitting, so
Table 4/6's existing numbers stay valid; this adds a parallel dataset.

Usage:
    python3 process_acl2019_grobid.py <pdf_dir> <out_dir> \
        [--grobid-url http://localhost:8070] [--workers 6]

Resumable: skips any <id> whose <out_dir>/<id>.tei.xml already exists.
Writes one <out_dir>/<id>.tei.xml per PDF, plus a fail log at
<out_dir>/failed.log (id + HTTP status/error, one per line).
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.error
import urllib.request

BOUNDARY = "----grobidBatchBoundary"


def post_pdf(grobid_url, pdf_path, timeout=120):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    body = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="input"; filename="{pdf_path.name}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + pdf_bytes + f"\r\n--{BOUNDARY}--\r\n".encode()
    req = urllib.request.Request(
        f"{grobid_url}/api/processFulltextDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def process_one(grobid_url, pdf_path, out_path):
    try:
        status, content = post_pdf(grobid_url, pdf_path)
    except urllib.error.HTTPError as e:
        return pdf_path.stem, False, f"HTTP {e.code}"
    except Exception as e:
        return pdf_path.stem, False, f"{type(e).__name__}: {e}"
    if status != 200 or not content.strip():
        return pdf_path.stem, False, f"status={status} empty={not content.strip()}"
    out_path.write_bytes(content)
    return pdf_path.stem, True, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf_dir")
    p.add_argument("out_dir")
    p.add_argument("--grobid-url", default="http://localhost:8070")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    todo = [pdf for pdf in pdfs if not (out_dir / f"{pdf.stem}.tei.xml").exists()]
    print(f"{len(pdfs)} PDFs, {len(todo)} to process "
          f"({len(pdfs) - len(todo)} already done)", file=sys.stderr)

    done = failed = 0
    start = time.time()
    fail_log = open(out_dir / "failed.log", "a")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, args.grobid_url, pdf, out_dir / f"{pdf.stem}.tei.xml"): pdf
            for pdf in todo
        }
        for i, future in enumerate(as_completed(futures), 1):
            doc_id, ok, error = future.result()
            if ok:
                done += 1
            else:
                failed += 1
                fail_log.write(f"{doc_id}\t{error}\n")
                fail_log.flush()
            if i % 100 == 0:
                elapsed = time.time() - start
                print(f"progress: {i}/{len(todo)} done={done} failed={failed} "
                      f"({elapsed:.0f}s, {i / elapsed:.1f}/s)", file=sys.stderr)

    print(f"DONE done={done} failed={failed} (of {len(todo)} attempted)", file=sys.stderr)


if __name__ == "__main__":
    main()

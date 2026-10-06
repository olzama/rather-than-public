#!/usr/bin/env python3
"""
Splice the blinded item sample into rather_than_annotator.html's
ITEMS_PLACEHOLDER, producing the file actually published as the
Artifact. Kept as an explicit build step (rather than hand-editing the
spliced output) so the page can be regenerated if the sample changes.

Usage:
    python3 build_page.py <items_public.jsonl> [out.html] [--revision revision/items_public.jsonl] \
        [--extra-items ../data/annotation_b3/items_public.jsonl ...]
Defaults out.html to rather_than_annotator_built.html next to this script.
--revision adds the revision set, which the page opens when its link ends
in "#revision". --extra-items appends items kept in another pool folder
(e.g. batch 3, analysed separately from the main pool). --sheet-url makes the page save to and load from a Google
Sheet through sheet_backend.gs (for hosting outside claude.ai).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out", nargs="?", default=str(HERE / "rather_than_annotator_built.html"))
    ap.add_argument("--revision", help="items_public.jsonl of the revision set")
    ap.add_argument("--extra-items", nargs="+", default=[], help="further items_public.jsonl files to append")
    ap.add_argument("--sheet-url", default="", help="Google Apps Script web app URL; saves go there instead of the claude.ai database")
    ap.add_argument("--sheet-token", default="rt-2026-annotate")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items)]
    for extra in args.extra_items:
        items += [json.loads(l) for l in open(extra)]
    if len({it["item_id"] for it in items}) != len(items):
        raise SystemExit("duplicate item_ids across item files")
    revision = [json.loads(l) for l in open(args.revision)] if args.revision else []
    template = (HERE / "rather_than_annotator.html").read_text()
    for ph in ("ITEMS_PLACEHOLDER", "REVISION_PLACEHOLDER", "SHEET_PLACEHOLDER"):
        if ph not in template:
            raise SystemExit(f"template is missing {ph}")
    page = template.replace("ITEMS_PLACEHOLDER", "const ITEMS = " + json.dumps(items) + ";")
    page = page.replace("REVISION_PLACEHOLDER", "const REVISION_ITEMS = " + json.dumps(revision) + ";")
    page = page.replace("SHEET_PLACEHOLDER", "const SHEET_URL = " + json.dumps(args.sheet_url) +
                        "; const SHEET_TOKEN = " + json.dumps(args.sheet_token) + ";")
    Path(args.out).write_text(page)
    print(f"wrote {args.out} ({len(items)} items, {len(revision)} revision items)")


if __name__ == "__main__":
    main()

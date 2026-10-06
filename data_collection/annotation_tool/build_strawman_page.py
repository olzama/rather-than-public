#!/usr/bin/env python3
"""
Splice the straw-man item sets (dev.jsonl, test.jsonl from strawman_sets.py)
into strawman_coder.html's SETS_PLACEHOLDER, producing the published page.

Usage:
    python3 build_strawman_page.py <strawman_dir> [out.html] [--sheet-url URL]
Defaults out.html to strawman_coder_built.html next to this script.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strawman_dir")
    ap.add_argument("out", nargs="?", default=str(HERE / "strawman_coder_built.html"))
    ap.add_argument("--sheet-url", default="", help="Google Apps Script web app URL (sheet_backend.gs); saves go there")
    ap.add_argument("--sheet-token", default="rt-2026-annotate")
    args = ap.parse_args()
    d = Path(args.strawman_dir)
    sets = {name: [json.loads(l) for l in open(d / f"{name}.jsonl")] for name in ("dev", "test")}
    template = (HERE / "strawman_coder.html").read_text()
    for ph in ("SETS_PLACEHOLDER", "SHEET_PLACEHOLDER"):
        if ph not in template:
            raise SystemExit(f"template is missing {ph}")
    page = template.replace("SETS_PLACEHOLDER", json.dumps(sets))
    page = page.replace("SHEET_PLACEHOLDER", "  const SHEET_URL = " + json.dumps(args.sheet_url) +
                        "; const SHEET_TOKEN = " + json.dumps(args.sheet_token) + ";")
    Path(args.out).write_text(page)
    print(f"wrote {args.out} (dev {len(sets['dev'])}, test {len(sets['test'])})")


if __name__ == "__main__":
    main()

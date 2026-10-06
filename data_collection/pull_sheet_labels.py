#!/usr/bin/env python3
r"""
Pull annotators' labels from the Google Sheet backend (sheet_backend.gs)
into the repo's label files. For each annotator the sheet returns the
latest label per item; that annotator's rows in the output files are
replaced, and other annotators' rows are kept.

  main items and "rep__" hidden repeats -> <annotation_dir>/human_labels.jsonl
  revision items ("rev_")               -> <annotation_dir>/revision/human_labels.jsonl

  straw-man codes ("sm:<set>:<item_id>", notes "smnote:...")
                                        -> <annotation_dir>/strawman/human_codes.jsonl
"ts" is the client timestamp of the latest row (null if the deployed
backend predates timestamp support).

Usage:
    python3 pull_sheet_labels.py ../data/annotation --sheet-url URL \
        --annotators A2 A5 [--token rt-2026-annotate] [--extra-pool ../data/annotation_b3]

With --extra-pool DIR, labels for items listed in DIR/items_public.jsonl
(and their "rep__" repeats) go to DIR/human_labels.jsonl instead.
"""
import argparse
import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path


def ssl_context():
    ctx = ssl.create_default_context()
    if not ctx.get_ca_certs():
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ctx


def fetch(url, token, name):
    q = urllib.parse.urlencode({"token": token, "annotator": name, "ts": 1})
    with urllib.request.urlopen(f"{url}?{q}", context=ssl_context(), timeout=60) as r:
        body = json.load(r)
    if "answers" not in body:
        raise SystemExit(f"{name}: {body}")
    ts = body.get("ts", {})
    return {k: (v, int(ts[k]) if ts.get(k) not in (None, "") else None) for k, v in body["answers"].items()}


def replace_rows(path, name, rows):
    old = [json.loads(l) for l in open(path)] if path.exists() else []
    kept = [r for r in old if r["annotator"].strip().lower() != name.lower()]
    # keep a known timestamp when the sheet returns none for an unchanged label
    key = lambda r: (r.get("set"), r["item_id"])
    value = lambda r: r.get("label", r.get("code"))
    prev = {key(r): r for r in old if r["annotator"].strip().lower() == name.lower()}
    for r in rows:
        p = prev.get(key(r))
        if r["ts"] is None and p and value(p) == value(r):
            r["ts"] = p.get("ts")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in kept + rows:
            f.write(json.dumps(r) + "\n")
    return len(old) - len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("--sheet-url", required=True)
    ap.add_argument("--token", default="rt-2026-annotate")
    ap.add_argument("--annotators", nargs="+", required=True)
    ap.add_argument("--extra-pool", nargs="+", default=[], type=Path,
                    help="pool folders whose items' labels go to their own human_labels.jsonl")
    args = ap.parse_args()
    d = Path(args.annotation_dir)
    extra_ids = {p: {json.loads(l)["item_id"] for l in open(p / "items_public.jsonl")} for p in args.extra_pool}
    for name in args.annotators:
        answers = fetch(args.sheet_url, args.token, name)
        main_rows, rev_rows, sm_rows = [], [], {}
        extra_rows = {pool: [] for pool in args.extra_pool}
        for item_id, (label, ts) in sorted(answers.items()):
            row = {"item_id": item_id, "annotator": name, "label": label, "ts": ts}
            if item_id.startswith("rev_"):
                rev_rows.append(row)
            elif item_id.startswith(("item_", "rep__")):
                base = item_id[len("rep__"):] if item_id.startswith("rep__") else item_id
                pool = next((p for p in args.extra_pool if base in extra_ids[p]), None)
                (extra_rows[pool] if pool else main_rows).append(row)
            elif item_id.startswith(("sm:", "smnote:")):
                kind, sm_set, sm_item = item_id.split(":", 2)
                r = sm_rows.setdefault((sm_set, sm_item), {"annotator": name, "set": sm_set, "item_id": sm_item,
                                                           "code": None, "note": "", "ts": ts})
                if kind == "sm":
                    r["code"], r["ts"] = label, ts
                else:
                    r["note"] = label
        sm_list = [r for r in sm_rows.values() if r["code"]]
        for pool, rows in extra_rows.items():
            if rows:
                dropped = replace_rows(pool / "human_labels.jsonl", name, rows)
                print(f"{name}: {len(rows)} rows -> {pool / 'human_labels.jsonl'} (replaced {dropped})")
        for path, rows in ((d / "human_labels.jsonl", main_rows), (d / "revision" / "human_labels.jsonl", rev_rows),
                           (d / "strawman" / "human_codes.jsonl", sm_list)):
            if rows:
                dropped = replace_rows(path, name, rows)
                print(f"{name}: {len(rows)} rows -> {path} (replaced {dropped})")


if __name__ == "__main__":
    main()

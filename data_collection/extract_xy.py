#!/usr/bin/env python3
r"""
Extract the two alternatives of each "rather than" item with an OpenAI model:
X, the alternative the sentence affirms, and Y, the one it rejects (in
"X rather than Y" and in sentence-initial "Rather than Y, X"). Both must be
exact substrings of the sentence (ignoring whitespace differences); answers
that are not are retried once and otherwise recorded with "valid": false.
X and Y are stored whitespace-normalized.

Hidden repeat items (repeats.jsonl) are skipped. Results are appended to
<out.jsonl>; items already there are skipped, so the run can be resumed.

The API key is read from $OPENAI_API_KEY, or from --key-file.

Usage:
    python3 extract_xy.py <items_public.jsonl> <out.jsonl> [--model gpt-6-sol] \
        [--key-file ../../LYS-API-key.txt] [--workers 8] [--limit N]
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm_judge as J

INSTRUCTIONS = """The sentence below contains one occurrence of "rather than", marked with [[ ]]. It contrasts two alternatives:
- X: the alternative the sentence affirms or prefers
- Y: the alternative it rejects

In "X rather than Y", X precedes and Y follows the marker; in sentence-initial "Rather than Y, X", Y comes first. Give X and Y as the shortest phrases that fully express each alternative, copied exactly (same characters) from the sentence, without the words "rather than" and without the brackets.

Sentence:
"""

SCHEMA = {
    "name": "alternatives",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"x": {"type": "string"}, "y": {"type": "string"}},
        "required": ["x", "y"],
        "additionalProperties": False,
    },
}


def ask(model, key, item):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": INSTRUCTIONS + J.marked_sentence(item)}],
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
    }
    if model.startswith(J.REASONING_PREFIXES):
        body["reasoning_effort"] = "none"
    else:
        body["temperature"] = 0
    resp = J.post(key, body)
    return json.loads(resp["choices"][0]["message"]["content"]), resp.get("usage", {}), resp.get("model", model)


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def extract(model, key, item):
    s = norm(item["sentence"])
    for attempt in range(2):
        out, usage, served = ask(model, key, item)
        x, y = norm(out["x"]), norm(out["y"])
        if x and y and x in s and y in s:
            break
    return {"item_id": item["item_id"], "model": served, "x": x, "y": y,
            "valid": bool(x and y and x in s and y in s), "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--model", default="gpt-6-sol")
    ap.add_argument("--key-file")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY") or (Path(args.key_file).read_text().strip() if args.key_file else None)
    if not key:
        sys.exit("no API key: set OPENAI_API_KEY or pass --key-file")

    items_path = Path(args.items)
    repeats = set()
    if (items_path.parent / "repeats.jsonl").exists():
        repeats = {json.loads(l)["item_id"] for l in open(items_path.parent / "repeats.jsonl")}
    items = sorted((it for it in map(json.loads, open(items_path)) if it["item_id"] not in repeats),
                   key=lambda it: it["item_id"])
    out = Path(args.out)
    done = {json.loads(l)["item_id"] for l in open(out)} if out.exists() else set()
    todo = [it for it in items if it["item_id"] not in done][:args.limit]
    print(f"{len(todo)} to extract ({len(done)} done)", file=sys.stderr)
    with open(out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(lambda it: extract(args.model, key, it), todo):
            f.write(json.dumps(res) + "\n")
            f.flush()
    print(f"DONE -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

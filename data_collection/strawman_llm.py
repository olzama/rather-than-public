#!/usr/bin/env python3
r"""
Code whether the rejected alternative (Y) of each item in a straw-man set
(dev.jsonl or test.jsonl from strawman_sets.py) is a straw man, with an
OpenAI model, using the same definition the human coders see
(definition.txt; keep it in sync with strawman_coder.html). The model sees
the sentence with "rather than" and Y marked, and the sentences before and
after; never the annoying/legitimate label.

Results are appended to <out_dir>/<set>__<model>__<reasoning>.jsonl (code:
strawman / not_strawman / unsure); items already there are skipped.

The API key is read from $OPENAI_API_KEY, or from --key-file.

Usage:
    python3 strawman_llm.py ../data/annotation/strawman/test.jsonl ../data/annotation/strawman/llm \
        --model gpt-6-astra [--reasoning none] [--definition ../data/annotation/strawman/definition.txt] \
        [--key-file ../../LYS-API-key.txt] [--workers 8] [--limit N]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm_judge as J

SCHEMA = {
    "name": "strawman_code",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "enum": ["strawman", "not_strawman", "unsure"]}},
        "required": ["code"],
        "additionalProperties": False,
    },
}


def marked(it):
    s, marks = it["sentence"], []
    if it.get("rt_start") is not None:
        marks.append((it["rt_start"], it["rt_end"], "[[", "]]"))
    if it.get("y_start") is not None:
        marks.append((it["y_start"], it["y_end"], "<<", ">>"))
    out, pos = "", 0
    for a, b, l, r in sorted(marks):
        if a < pos:
            continue
        out += s[pos:a] + l + s[a:b] + r
        pos = b
    return out + s[pos:]


def prompt(definition, it):
    parts = [p for p in (it.get("context_before", ""), marked(it), it.get("context_after", "")) if p and p.strip()]
    return (
        "You will see an excerpt from a scientific paper. In the middle sentence, [[ ]] marks \"rather than\" "
        "and << >> marks the alternative it rejects.\n\n"
        f"Definition:\n{definition}\n\n"
        "Is the alternative marked << >> a straw man? Answer strawman, not_strawman, or unsure.\n\n"
        "Excerpt:\n" + "\n\n".join(parts)
    )


def code(model, key, reasoning, definition, it):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt(definition, it)}],
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
    }
    if model.startswith(J.REASONING_PREFIXES):
        body["reasoning_effort"] = reasoning
    else:
        body["temperature"] = 0
    resp = J.post(key, body)
    out = json.loads(resp["choices"][0]["message"]["content"])
    return {"item_id": it["item_id"], "model": resp.get("model", model), "reasoning": reasoning,
            "code": out["code"], "usage": resp.get("usage", {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_file")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--definition", default=None)
    ap.add_argument("--key-file")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY") or (Path(args.key_file).read_text().strip() if args.key_file else None)
    if not key:
        sys.exit("no API key: set OPENAI_API_KEY or pass --key-file")
    set_path = Path(args.set_file)
    definition = Path(args.definition or set_path.parent / "definition.txt").read_text().strip()
    items = [json.loads(l) for l in open(set_path)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{set_path.stem}__{args.model}__{args.reasoning}.jsonl"
    done = {json.loads(l)["item_id"] for l in open(out)} if out.exists() else set()
    todo = [it for it in items if it["item_id"] not in done][:args.limit]
    print(f"{out.name}: {len(todo)} to code ({len(done)} done)", file=sys.stderr)
    with open(out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(lambda it: code(args.model, key, args.reasoning, definition, it), todo):
            f.write(json.dumps(res) + "\n")
            f.flush()
    print(f"DONE -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

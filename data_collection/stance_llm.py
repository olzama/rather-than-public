#!/usr/bin/env python3
r"""
Evaluative stance toward the two alternatives of each "rather than" item,
scored by an OpenAI model: how favourably the author presents X (the
alternative affirmed) and Y (the one rejected), each from -2 (clearly
disparaged) to +2 (clearly favoured). The model sees the sentence with X
and Y marked and the sentences before and after; never the annotators'
labels or straw-man codes. Anchoring examples are invented (not from the
data).

Input: items_public.jsonl and xy_alternatives.jsonl (extract_xy.py; items
without a valid X/Y are skipped). Results are appended to --out (item_id,
x, y, model); items already there are skipped.

Usage:
    python3 stance_llm.py ../data/annotation ../data/annotation/stance/stance__gpt-6-sol.jsonl \
        --key-file ../../LYS-API-key.txt [--model gpt-6-sol] [--reasoning low] [--workers 8] [--limit N]
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm_judge as J

INSTRUCTIONS = """The sentence below is from a scientific paper. It contrasts two alternatives with "rather than": X, marked <<X: ...>>, which the sentence affirms, and Y, marked <<Y: ...>>, which it rejects.

Rate how the author presents each alternative, from the wording of the passage alone (not from your own view of the methods):
 -2 = clearly disparaged (presented as naive, flawed, careless, or bad)
 -1 = somewhat negative
  0 = neutral (presented as a plain option, without evaluation)
 +1 = somewhat positive
 +2 = clearly favoured (presented as principled, meaningful, or good)

Invented examples:
"We tune on the development set <<Y: rather than on the test set>>" -> X 0, Y -1
"The model learns <<X: meaningful structure>> rather than <<Y: superficial artifacts>>" -> X +2, Y -2
"We use <<X: a CRF>> rather than <<Y: a softmax layer>>" -> X 0, Y 0
"""

SCHEMA = {"name": "stance", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["x", "y"],
    "properties": {"x": {"type": "integer", "enum": [-2, -1, 0, 1, 2]},
                   "y": {"type": "integer", "enum": [-2, -1, 0, 1, 2]}}}}


def mark(sentence, x, y):
    def find(s, phrase):
        m = re.search(r"\s+".join(re.escape(w) for w in phrase.split()), s)
        return (m.start(), m.end()) if m else None
    spans = [(find(sentence, x), "X"), (find(sentence, y), "Y")]
    if any(sp is None for sp, _ in spans):
        return None
    spans.sort(key=lambda t: t[0][0])
    (a, b), la = spans[0]
    (c, d), lb = spans[1]
    if c < b:
        return None
    return (sentence[:a] + f"<<{la}: " + sentence[a:b] + ">>" + sentence[b:c] + f"<<{lb}: " + sentence[c:d] + ">>"
            + sentence[d:])


def score(model, key, reasoning, item, marked):
    parts = [p for p in (item.get("context_before", ""), marked, item.get("context_after", "")) if p and p.strip()]
    body = {"model": model,
            "messages": [{"role": "user", "content": INSTRUCTIONS + "\nPassage:\n" + "\n".join(" ".join(p.split()) for p in parts)}],
            "response_format": {"type": "json_schema", "json_schema": SCHEMA}}
    if model.startswith(J.REASONING_PREFIXES):
        body["reasoning_effort"] = reasoning
    resp = J.post(key, body)
    out = json.loads(resp["choices"][0]["message"]["content"])
    return {"item_id": item["item_id"], "x": out["x"], "y": out["y"], "model": resp.get("model", model),
            "usage": resp.get("usage", {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("out")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model", default="gpt-6-sol")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    key = Path(args.key_file).read_text().strip()
    d = Path(args.annotation_dir)
    pub = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl"))}
    xy = {r["item_id"]: r for r in map(json.loads, open(d / "xy_alternatives.jsonl")) if r.get("valid") and r.get("x") and r.get("y")}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["item_id"] for l in open(out)} if out.exists() else set()
    jobs = []
    for i in sorted(xy):
        if i in pub and i not in done:
            m = mark(" ".join(pub[i]["sentence"].split()), xy[i]["x"], xy[i]["y"])
            if m:
                jobs.append((pub[i], m))
    jobs = jobs[:args.limit]
    print(f"{out.name}: {len(jobs)} to score ({len(done)} done)", file=sys.stderr)
    with open(out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        for r in ex.map(lambda j: score(args.model, key, args.reasoning, j[0], j[1]), jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()


if __name__ == "__main__":
    main()

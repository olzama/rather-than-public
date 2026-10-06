#!/usr/bin/env python3
r"""
Sentiment of each text in the reward-model test, scored by an OpenAI model:
the original sentence, its length-matched control and its deleted version
(pairs_control.jsonl from build_length_controls.py). Each text is rated on its
own, in random order across items, without context and without being told
which version it is or that the versions are related. Rating: the overall
evaluative tone of the sentence toward what it describes, from -2 (clearly
negative) to +2 (clearly positive), 0 = neutral/descriptive.

Output: JSONL, one line per text: item_id, text (original|control|deleted),
llm (the rating), model. Existing lines are skipped, so reruns resume.

Usage:
    python3 sentiment_llm.py ../data/reward_pairs/pairs_control.jsonl ../data/reward_pairs/sentiment.jsonl \
        --key-file <key file> [--model gpt-6-sol] [--reasoning low] [--workers 8] [--limit N]
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm_judge as J

INSTRUCTIONS = """The sentence below is from a scientific paper. Rate its overall sentiment: the evaluative tone with which it presents what it describes (methods, results, claims, other work), judged from the wording alone, not from your own view of the content.
 -2 = clearly negative (criticism, failure, flaws, limitations)
 -1 = somewhat negative
  0 = neutral (descriptive, no evaluation)
 +1 = somewhat positive
 +2 = clearly positive (praise, success, strengths, advantages)

Invented examples:
"We tune the hyperparameters on the development set." -> 0
"Our method substantially improves accuracy and is remarkably robust." -> +2
"The baseline fails on long inputs and its outputs are often incoherent." -> -2

Sentence:
"""

SCHEMA = {"name": "sentiment", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["rating"],
    "properties": {"rating": {"type": "integer", "enum": [-2, -1, 0, 1, 2]}}}}


def score(model, key, reasoning, item_id, kind, text):
    body = {"model": model,
            "messages": [{"role": "user", "content": INSTRUCTIONS + " ".join(text.split())}],
            "response_format": {"type": "json_schema", "json_schema": SCHEMA}}
    if model.startswith(J.REASONING_PREFIXES):
        body["reasoning_effort"] = reasoning
    resp = J.post(key, body)
    out = json.loads(resp["choices"][0]["message"]["content"])
    return {"item_id": item_id, "text": kind, "llm": out["rating"], "model": resp.get("model", model),
            "usage": resp.get("usage", {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model", default="gpt-6-sol")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    key = Path(args.key_file).read_text().strip()
    out = Path(args.out)
    done = {(r["item_id"], r["text"]) for r in map(json.loads, open(out))} if out.exists() else set()
    jobs = []
    for p in map(json.loads, open(args.pairs)):
        for kind, text in (("original", p["true_original"]), ("control", p["original"]), ("deleted", p["deleted"])):
            if (p["item_id"], kind) not in done:
                jobs.append((p["item_id"], kind, text))
    random.Random(0).shuffle(jobs)
    jobs = jobs[:args.limit]
    print(f"{out.name}: {len(jobs)} to score ({len(done)} done)", file=sys.stderr)
    with open(out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        for r in ex.map(lambda j: score(args.model, key, args.reasoning, *j), jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()


if __name__ == "__main__":
    main()

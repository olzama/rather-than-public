#!/usr/bin/env python3
r"""
Ask an OpenAI model to judge each "rather than" item in the annotation pool.
The model sees what the human annotator sees: the sentence with the
occurrence marked, and the sentence before and after. Three prompts:

  generic   How annoying would a reader find this use? 1-5 score plus a
            label (legitimate / annoying / unsure / garbled).
  deletion  Would deleting "rather than X" lose anything? 1-5 score
            (1 = loses essential information, 5 = loses nothing) plus a
            label (needed / redundant / unsure / garbled).
  fewshot   The generic prompt, preceded by labeled examples from one human
            annotator: --shots annoying and --shots legitimate items drawn
            at random (--seed) from that annotator's labels. Those example
            items are not judged; the drawn split is saved next to the output.

Hidden repeat items (repeats.jsonl) are skipped. Results are appended one
JSON object per line to <out_dir>/<model>__<prompt>__<reasoning>.jsonl; items
already present there are skipped, so an interrupted run can be resumed by
rerunning the command.

The API key is read from $OPENAI_API_KEY, or from --key-file.

Usage:
    python3 llm_judge.py <items_public.jsonl> <out_dir> --model gpt-6-sol \
        [--prompt generic|deletion|fewshot] [--reasoning none] \
        [--labels human_labels.jsonl --annotator A1 --shots 20 --seed 1] \
        [--key-file ../../LYS-API-key.txt] [--limit 20] [--workers 8]
"""
import argparse
import json
import os
import random
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = (
    "You are a reader of scientific papers. You will see a short excerpt from a "
    "paper. One occurrence of \"rather than\" in the middle sentence is marked "
    "with [[ ]]. Judge only that marked occurrence."
)

GENERIC = """How annoying would a reader find the marked use of "rather than"?

Give a score from 1 to 5:
1 = not annoying at all, 2 = barely, 3 = somewhat, 4 = annoying, 5 = very annoying.

Also give one label:
- legitimate: the marked use does not annoy a reader
- annoying: the marked use annoys a reader
- unsure: you cannot tell
- garbled: the text is too broken by extraction to read
"""

DELETION = """Suppose the marked "rather than" and the alternative it introduces were deleted from the sentence (with minimal edits so it still reads well). Would the sentence lose anything a reader needs?

Give a score from 1 to 5:
1 = loses essential information, 2 = loses something useful, 3 = loses a little, 4 = loses almost nothing, 5 = loses nothing.

Also give one label:
- needed: deleting it would lose something a reader needs
- redundant: deleting it would lose nothing a reader needs
- unsure: you cannot tell
- garbled: the text is too broken by extraction to read
"""

LABELS = {
    "generic": ["legitimate", "annoying", "unsure", "garbled"],
    "deletion": ["needed", "redundant", "unsure", "garbled"],
    "fewshot": ["legitimate", "annoying", "unsure", "garbled"],
}

REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")

# python.org builds of Python on macOS ship without root certificates;
# fall back to the system bundle when the default store is empty.
SSL_CONTEXT = ssl.create_default_context()
if not SSL_CONTEXT.get_ca_certs() and Path("/etc/ssl/cert.pem").exists():
    SSL_CONTEXT = ssl.create_default_context(cafile="/etc/ssl/cert.pem")


def marked_sentence(item):
    s = item["sentence"]
    return s[:item["local_start"]] + "[[" + s[item["local_start"]:item["local_end"]] + "]]" + s[item["local_end"]:]


def excerpt(item):
    parts = [item.get("context_before", ""), marked_sentence(item), item.get("context_after", "")]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def schema(prompt):
    return {
        "name": "judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "label": {"type": "string", "enum": LABELS[prompt]},
            },
            "required": ["score", "label"],
            "additionalProperties": False,
        },
    }


def build_instructions(prompt, examples):
    if prompt == "deletion":
        return DELETION
    if prompt == "generic":
        return GENERIC
    lines = [GENERIC, "Here are examples of how one reader labeled other marked uses:\n"]
    for it, label in examples:
        lines.append(f"- {label}: {marked_sentence(it).strip()}")
    return "\n".join(lines) + "\n"


def post(key, body, endpoint="chat/completions", retries=5):
    """POST to the OpenAI API with retries on rate limits and transient errors."""
    req = urllib.request.Request(
        f"https://api.openai.com/v1/{endpoint}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code} {e.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def call(model, key, item, prompt, instructions, reasoning):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": instructions + "\nExcerpt:\n" + excerpt(item)},
        ],
        "response_format": {"type": "json_schema", "json_schema": schema(prompt)},
    }
    if model.startswith(REASONING_PREFIXES):
        body["reasoning_effort"] = reasoning
    else:
        body["temperature"] = 0
    resp = post(key, body)
    out = json.loads(resp["choices"][0]["message"]["content"])
    return {
        "item_id": item["item_id"], "model": resp.get("model", model),
        "prompt": prompt, "reasoning": reasoning,
        "score": out["score"], "label": out["label"],
        "usage": resp.get("usage", {}),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("items")
    p.add_argument("out_dir")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", choices=sorted(LABELS), default="generic")
    p.add_argument("--reasoning", default="none",
                   help="reasoning_effort for reasoning models (e.g. none, minimal, low); ignored otherwise")
    p.add_argument("--labels", help="human_labels.jsonl, for --prompt fewshot")
    p.add_argument("--annotator", default="A1", help="whose labels supply the examples, for --prompt fewshot")
    p.add_argument("--shots", type=int, default=20, help="examples per class, for --prompt fewshot")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--key-file")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key and args.key_file:
        key = Path(args.key_file).read_text().strip()
    if not key:
        sys.exit("no API key: set OPENAI_API_KEY or pass --key-file")

    items_path = Path(args.items)
    repeats_path = items_path.parent / "repeats.jsonl"
    repeats = set()
    if repeats_path.exists():
        repeats = {json.loads(l)["item_id"] for l in open(repeats_path)}
    items = [json.loads(l) for l in open(items_path)]
    items = sorted((it for it in items if it["item_id"] not in repeats), key=lambda it: it["item_id"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}__{args.prompt}__{args.reasoning}"

    examples = []
    if args.prompt == "fewshot":
        if not args.labels:
            sys.exit("--prompt fewshot needs --labels")
        human = {}
        for l in open(args.labels):
            r = json.loads(l)
            if r["annotator"] == args.annotator:
                human[r["item_id"]] = r["label"]
        rng = random.Random(args.seed)
        by_id = {it["item_id"]: it for it in items}
        for label in ("annoying", "legitimate"):
            pool = sorted(i for i in by_id if human.get(i) == label)
            examples += [(by_id[i], label) for i in rng.sample(pool, args.shots)]
        rng.shuffle(examples)
        tag += f"__{args.shots}x2-seed{args.seed}"
        shot_ids = {it["item_id"] for it, _ in examples}
        (out_dir / f"{tag}.examples.json").write_text(json.dumps(sorted(shot_ids), indent=1))
        items = [it for it in items if it["item_id"] not in shot_ids]
    instructions = build_instructions(args.prompt, examples)

    out_path = out_dir / f"{tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["item_id"] for l in open(out_path)}
    todo = [it for it in items if it["item_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{tag}: {len(todo)} to judge ({len(done)} already done)", file=sys.stderr)

    n = 0
    with open(out_path, "a") as out, ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(lambda it: call(args.model, key, it, args.prompt, instructions, args.reasoning), todo):
            out.write(json.dumps(res) + "\n")
            out.flush()
            n += 1
    print(f"DONE {n} -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

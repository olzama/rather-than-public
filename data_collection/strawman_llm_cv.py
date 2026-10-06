#!/usr/bin/env python3
r"""
Few-shot LLM straw-man coder, evaluated by cross-validation against a
human coder's codes.

The human-coded items (dev.jsonl and test.jsonl from strawman_sets.py, with
the coder's codes in human_codes.jsonl) are split into --folds folds,
stratified by code. Each fold is coded by the model with --shots straw-man
and --shots not-straw-man examples drawn from the other folds (the same
definition as strawman_llm.py, examples shown as excerpt + code), so no item
is coded with itself as an example. Agreement with the human codes is
reported as Cohen's kappa on straw man vs. other, pooled over folds, with
the human coder's own agreement with other coders where they overlap.

Results are appended to <out_dir>/cv__<model>__<reasoning>__<shots>shot.jsonl;
items already there are skipped.

Usage:
    python3 strawman_llm_cv.py ../data/annotation/strawman --coder A1 --model gpt-6-sol \
        [--reasoning low] [--shots 8] [--folds 5] [--seed 1] [--key-file ../../LYS-API-key.txt]
"""
import argparse
import collections
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm_judge as J
from strawman_llm import SCHEMA, marked


def excerpt(it):
    return "\n\n".join(p for p in (it.get("context_before", ""), marked(it), it.get("context_after", "")) if p and p.strip())


def prompt(definition, examples, it):
    shots = "\n\n".join(f"Example excerpt:\n{excerpt(e)}\nCode: {c}" for e, c in examples)
    return (
        "You will see an excerpt from a scientific paper. In the middle sentence, [[ ]] marks \"rather than\" "
        "and << >> marks the alternative it rejects.\n\n"
        f"Definition:\n{definition}\n\n"
        f"Examples coded by a human coder with this definition:\n\n{shots}\n\n"
        "Is the alternative marked << >> in the following excerpt a straw man? "
        "Answer strawman, not_strawman, or unsure.\n\n"
        "Excerpt:\n" + excerpt(it)
    )


def code(model, key, reasoning, definition, examples, it, fold):
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt(definition, examples, it)}],
            "response_format": {"type": "json_schema", "json_schema": SCHEMA}}
    if model.startswith(J.REASONING_PREFIXES):
        body["reasoning_effort"] = reasoning
    else:
        body["temperature"] = 0
    resp = J.post(key, body)
    out = json.loads(resp["choices"][0]["message"]["content"])
    return {"item_id": it["item_id"], "fold": fold, "code": out["code"], "model": resp.get("model", model),
            "reasoning": reasoning, "examples": [e["item_id"] for e, _ in examples], "usage": resp.get("usage", {})}


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = collections.Counter(x), collections.Counter(y)
    pe = sum(cx[v] * cy[v] for v in set(x) | set(y)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strawman_dir")
    ap.add_argument("--coder", default="A1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--key-file")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY") or (Path(args.key_file).read_text().strip() if args.key_file else None)
    if not key:
        sys.exit("no API key: set OPENAI_API_KEY or pass --key-file")
    d = Path(args.strawman_dir)
    definition = (d / "definition.txt").read_text().strip()
    items = {}
    for name in ("dev", "test"):
        for it in map(json.loads, open(d / f"{name}.jsonl")):
            items[it["item_id"]] = it
    codes = collections.defaultdict(dict)
    for r in map(json.loads, open(d / "human_codes.jsonl")):
        codes[r["annotator"]][r["item_id"]] = r["code"]
    human = {i: c for i, c in codes[args.coder].items() if i in items}

    rng = random.Random(args.seed)
    fold_of = {}
    for c in sorted(set(human.values())):
        ids = sorted(i for i in human if human[i] == c)
        rng.shuffle(ids)
        for k, i in enumerate(ids):
            fold_of[i] = k % args.folds

    out = d / "llm" / f"cv__{args.model}__{args.reasoning}__{args.shots}shot.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["item_id"] for l in open(out)} if out.exists() else set()
    jobs = []
    for f in range(args.folds):
        pool = [i for i in human if fold_of[i] != f]
        frng = random.Random(args.seed * 100 + f)
        ex = [(items[i], "strawman") for i in frng.sample(sorted(i for i in pool if human[i] == "strawman"), args.shots)]
        ex += [(items[i], "not_strawman") for i in frng.sample(sorted(i for i in pool if human[i] == "not_strawman"), args.shots)]
        frng.shuffle(ex)
        jobs += [(items[i], f, ex) for i in sorted(human) if fold_of[i] == f and i not in done]
    print(f"{out.name}: {len(jobs)} to code ({len(done)} done)", file=sys.stderr)
    with open(out, "a") as fh, ThreadPoolExecutor(args.workers) as pool:
        for res in pool.map(lambda j: code(args.model, key, args.reasoning, definition, j[2], j[0], j[1]), jobs):
            fh.write(json.dumps(res) + "\n")
            fh.flush()

    llm = {r["item_id"]: r["code"] for r in map(json.loads, open(out))}
    common = sorted(i for i in human if i in llm)
    h = [human[i] == "strawman" for i in common]
    m = [llm[i] == "strawman" for i in common]
    tokens = [0, 0]
    for r in map(json.loads, open(out)):
        tokens[0] += r["usage"].get("prompt_tokens", 0)
        tokens[1] += r["usage"].get("completion_tokens", 0)
    print(f"\n{args.coder} vs {args.model} ({args.reasoning}, {args.shots}+{args.shots} shots, {args.folds}-fold): "
          f"n={len(common)}; human straw men {sum(h)}, model {sum(m)}, both {sum(a and b for a, b in zip(h, m))}; "
          f"kappa {kappa(h, m):.2f}; tokens in/out {tokens[0]}/{tokens[1]}")
    print("confusion (human -> model):", dict(collections.Counter((human[i], llm[i]) for i in common)))
    for other, oc in codes.items():
        if other == args.coder:
            continue
        both = sorted(i for i in oc if i in human)
        if len(both) >= 5:
            print(f"human agreement {args.coder} vs {other}: n={len(both)}, kappa "
                  f"{kappa([human[i] == 'strawman' for i in both], [oc[i] == 'strawman' for i in both]):.2f}; "
                  f"model vs {other}: kappa {kappa([llm.get(i) == 'strawman' for i in both], [oc[i] == 'strawman' for i in both]):.2f}")


if __name__ == "__main__":
    main()

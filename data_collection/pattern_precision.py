#!/usr/bin/env python3
r"""
Precision of the antithesis patterns (antithesis_patterns.py): for a random
sample of matches per pattern and corpus, an OpenAI model judges whether the
matched text is an instance of the construction the pattern stands for.

Matches are drawn from the instance files (extract_antithesis_instances.py)
and kept only if they lie in the body text used for the rates (the body file
is a prefix of the full text, so a match is in the body when its offset is
below the body length); excluded documents are left out. Each judgment sees
the sentence, with the match marked <<...>>, and one sentence of context on
each side. Results are appended to --out (resumable).

Usage:
    python3 pattern_precision.py ../data/antithesis_instances ../data \
        ../data/pattern_precision/judgments.jsonl --key-file ../../LYS-API-key.txt \
        [--per-cell 50] [--seed 20260930] [--model gpt-6-sol] [--workers 8]
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm_judge as J
from antithesis_patterns import PATTERNS
from excluded_docs import drop_excluded

CORPORA = ("acl2019", "arxiv2026")

GENERAL = """The passage below is from a scientific paper. A pattern matcher flagged the text marked <<...>> as a possible instance of an antithesis construction: a contrast in which one alternative is negated, rejected or set aside in favour of another. Decide whether the marked text really is an instance of the construction described below. Answer "no" if the match is a coincidence of words, if the text is too garbled to read as the construction, or if it is part of a reference entry or a title."""

CRITERIA = {
    "rather_than": 'Construction: "X rather than Y", where Y is an alternative set aside in favour of X (including "would rather X than Y").',
    "instead_of": 'Construction: "X instead of Y", where Y is replaced by or set aside in favour of X.',
    "as_opposed_to": 'Construction: "X as opposed to Y", contrasting X with an alternative Y.',
    "in_contrast": 'Construction: "in contrast (to Y)" or "In contrast, ...", marking a contrast between two things, claims or findings.',
    "not_but": 'Construction: "not X but Y", where "but" introduces what replaces or corrects the negated X ("not a bug but a feature"). Answer "no" when "but" starts a concessive or adversative clause that does not replace X ("the results are not significant, but the trend is clear").',
    "x_not_y": 'Construction: postposed negation "X, not Y", where ", not Y" rejects an alternative to X stated just before ("we segment into words, not characters"). Answer "no" for other uses of a comma before "not" ("whether or not", "if not", ", not shown", participles such as ", not knowing").',
    "cleft_pivot": 'Construction: "it\'s not X, it\'s Y" / "it is not X; it is Y", denying X and asserting Y about the same thing.',
    "not_only_just_merely_but": 'Construction: "not only/just/merely X but (also) Y", pairing X with an added or emphasized Y.',
    "not_to_say": 'Construction: "this is not to say ...", a disclaimer that rejects an inference the reader might draw.',
}

SCHEMA = {"name": "match", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["instance"],
    "properties": {"instance": {"type": "string", "enum": ["yes", "no"]}}}}


def judge(model, key, reasoning, rec):
    s = rec["sentence"]
    marked = s[:rec["local_start"]] + "<<" + s[rec["local_start"]:rec["local_end"]] + ">>" + s[rec["local_end"]:]
    parts = [p for p in (rec.get("context_before", ""), marked, rec.get("context_after", "")) if p and p.strip()]
    body = {"model": model, "messages": [{"role": "user", "content":
            GENERAL + "\n\n" + CRITERIA[rec["pattern_id"]] + "\n\nPassage:\n" + "\n".join(" ".join(p.split()) for p in parts)}],
            "response_format": {"type": "json_schema", "json_schema": SCHEMA}}
    if model.startswith(J.REASONING_PREFIXES):
        body["reasoning_effort"] = reasoning
    resp = J.post(key, body)
    out = json.loads(resp["choices"][0]["message"]["content"])
    return {"key": rec["_key"], "corpus": rec["corpus"], "doc_id": rec["doc_id"], "pattern_id": rec["pattern_id"],
            "match_text": rec["match_text"], "sentence": rec["sentence"], "instance": out["instance"],
            "model": resp.get("model", model), "usage": resp.get("usage", {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instances_dir")
    ap.add_argument("data_dir")
    ap.add_argument("out")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--model", default="gpt-6-sol")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    key = Path(args.key_file).read_text().strip()
    rng = random.Random(args.seed)
    jobs = []
    for corpus in CORPORA:
        recs = [json.loads(l) for l in open(Path(args.instances_dir) / f"{corpus}.jsonl")]
        recs = drop_excluded(recs)
        body_len, full = {}, {}
        for r in recs:
            if r["doc_id"] not in body_len:
                f = Path(args.data_dir) / corpus / "text_body" / f"{r['doc_id']}.txt"
                g = Path(args.data_dir) / corpus / "text" / f"{r['doc_id']}.txt"
                body_len[r["doc_id"]] = len(f.read_text(errors="replace")) if f.exists() else -1
                full[r["doc_id"]] = g.read_text(errors="replace") if g.exists() else ""
        by, stale = {}, 0
        for r in recs:
            if full[r["doc_id"]][r["char_start"]:r["char_end"]].lower() != r["match_text"].lower():
                stale += 1  # offsets no longer match the current text file
                continue
            if 0 <= r["char_start"] < body_len[r["doc_id"]]:
                by.setdefault(r["pattern_id"], []).append(r)
        print(f"{corpus}: {stale} instances with stale offsets skipped; in body: "
              + ", ".join(f"{k} {len(v)}" for k, v in sorted(by.items())), file=sys.stderr)
        for pid, *_ in PATTERNS:
            pool = sorted(by.get(pid, []), key=lambda r: (r["doc_id"], r["char_start"]))
            for r in rng.sample(pool, min(args.per_cell, len(pool))):
                r["_key"] = f"{corpus}|{r['doc_id']}|{r['char_start']}|{pid}"
                jobs.append(r)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in open(out)} if out.exists() else set()
    jobs = [j for j in jobs if j["_key"] not in done]
    print(f"{len(jobs)} to judge ({len(done)} done)", file=sys.stderr)
    with open(out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        for r in ex.map(lambda j: judge(args.model, key, args.reasoning, j), jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Label the research domain of each arXiv 2026 paper behind the annotation
pool, from its title and abstract, with an OpenAI model. Categories are
fixed in advance (see counting/domain_test.py for the test they feed):

  core_nlp        the paper's question is about NLP/ML itself: models,
                  methods, training, evaluation, benchmarks, general-purpose
                  datasets, LLM behavior, safety, agents, multilingual or
                  low-resource NLP methods
  applied_domain  NLP applied to a specific non-NLP field whose questions or
                  data are central: medicine and health, mental health, law,
                  education, finance, social and political science, the
                  natural sciences, the humanities (other than linguistics)
  linguistics     the paper's main question is about language itself:
                  linguistic theory, typology, psycholinguistics, language
                  acquisition, sociolinguistics, the grammar of particular
                  languages, even when answered with computational methods

Title and abstract come from ../data/arxiv2026/sections.jsonl.gz (the
section labeled "abstract"; the first 250 words of the paper when there is
none). v1/latest items are mapped to their paper id. Output: one JSON line
per paper {doc_id, domain, field, model}; existing lines are kept, so the
script resumes.

Usage:
    python3 paper_domain.py ../data/annotation ../data/arxiv2026/sections.jsonl.gz \
        ../data/annotation/paper_domains.jsonl --key-file ../../LYS-API-key.txt \
        [--model gpt-6-sol] [--reasoning low]
"""
import argparse
import gzip
import json
import re
from pathlib import Path

from llm_judge import post

ARXIV = {"arxiv2026", "arxiv_v1", "arxiv_latest", "high_count_arxiv2026"}
VERSION_RE = re.compile(r"_v\d+$")

INSTRUCTIONS = """Classify the research domain of this NLP paper from its title and abstract.

core_nlp: the paper's question is about NLP/ML itself: models, methods, training, evaluation, benchmarks, general-purpose datasets, LLM behavior, safety, agents, multilingual or low-resource NLP methods.
applied_domain: NLP applied to a specific non-NLP field whose questions or data are central: medicine and health, mental health, law, education, finance, social and political science, the natural sciences, the humanities (other than linguistics).
linguistics: the paper's main question is about language itself: linguistic theory, typology, psycholinguistics, language acquisition, sociolinguistics, the grammar of particular languages, even when answered with computational methods.

Also name the field in a few words (e.g. "clinical NLP", "LLM evaluation", "typology")."""

SCHEMA = {"name": "domain", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["domain", "field"],
    "properties": {"domain": {"type": "string", "enum": ["core_nlp", "applied_domain", "linguistics"]},
                   "field": {"type": "string"}}}}


def paper_text(record):
    abstract = " ".join(s["text"] for s in record["sections"] if s["category"] == "abstract").strip()
    if not abstract:
        abstract = " ".join(" ".join(s["text"] for s in record["sections"]).split()[:250])
    return f"Title: {record.get('title', '')}\n\nAbstract: {abstract}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_dir")
    ap.add_argument("sections")
    ap.add_argument("out")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model", default="gpt-6-sol")
    ap.add_argument("--reasoning", default="low")
    args = ap.parse_args()
    key = open(args.key_file).read().strip()
    d = Path(args.annotation_dir)
    papers = sorted({VERSION_RE.sub("", r["doc_id"]) for r in map(json.loads, open(d / "items_provenance.jsonl"))
                     if r["corpus"] in ARXIV and "repeat_of" not in r and "excluded" not in r})
    wanted = set(papers)
    records = {}
    for line in gzip.open(args.sections, "rt"):
        r = json.loads(line)
        if r["doc_id"] in wanted:
            records[r["doc_id"]] = r
    done = {json.loads(l)["doc_id"] for l in open(args.out)} if Path(args.out).exists() else set()
    missing = [p for p in papers if p not in records]
    print(f"{len(papers)} papers; {len(done)} already labeled; no sections record for {len(missing)}: {missing[:10]}")
    usage = [0, 0]
    with open(args.out, "a") as out:
        for i, p in enumerate(p for p in papers if p in records and p not in done):
            body = {"model": args.model,
                    "messages": [{"role": "user", "content": INSTRUCTIONS + "\n\n" + paper_text(records[p])}],
                    "response_format": {"type": "json_schema", "json_schema": SCHEMA}}
            if args.reasoning:
                body["reasoning_effort"] = args.reasoning
            resp = post(key, body)
            ans = json.loads(resp["choices"][0]["message"]["content"])
            usage[0] += resp["usage"]["prompt_tokens"]
            usage[1] += resp["usage"]["completion_tokens"]
            out.write(json.dumps({"doc_id": p, **ans, "model": args.model}) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"{i} labeled; tokens in/out {usage[0]}/{usage[1]}")
    print(f"done; tokens in/out {usage[0]}/{usage[1]}")


if __name__ == "__main__":
    main()

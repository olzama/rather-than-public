#!/usr/bin/env python3
r"""
Items for the stance rating of the open-model and Claude pilots: every "rather than" instance in the pilot source
papers, from the originals and from each model's first drafts and revised papers, as <out_dir>/items_public.jsonl
(the input of extract_xy.py and stance_llm.py, which then write xy_alternatives.jsonl and stance__<model>.jsonl).

Each --source gives the instance file (extract_antithesis_instances.py, run on the papers of one model and stage) as
NAME:STAGE=INSTANCES.jsonl. A source whose STAGE is "any" holds several stages in its doc_id as
<model>__<stage>__<paper id> (the GPT files in data/antithesis_instances/). The originals are given as
original:original=INSTANCES.jsonl and cut at the end of the body text, as in the rates: --body-dir holds the body text
<id>.txt of the original papers, and instances that start after its end (references, appendices) are dropped.
Only pattern rather_than, only the papers of --ids-file (corpus<TAB>doc_id), an instance once (same source, stage,
paper and character offset). item_id = <source>|<stage>|<doc_id>|<char_start>.

Usage:
    python3 build_stance_pilot_items.py ../data/stance_pilot --ids-file ../data/generated/pilot_ids.tsv \
        --body-dir ../data/acl2019/text_body --body-dir ../data/arxiv2026/text_body \
        --source original:original=../data/antithesis_instances/acl2019.jsonl \
        --source gpt-6-sol:any=../data/antithesis_instances/llm_gpt-6-sol.jsonl \
        --source claude-opus-5-5:generated=INST/opus_generated.jsonl ...
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--source", action="append", required=True, metavar="NAME:STAGE=FILE")
    ap.add_argument("--body-dir", action="append", default=[])
    args = ap.parse_args()
    ids = {l.split()[1] for l in open(args.ids_file) if l.strip() and not l.startswith("#")}
    body_len = {p.stem: len(p.read_text(errors="replace")) for d in args.body_dir for p in Path(d).glob("*.txt")}
    items, seen = [], set()
    for spec in args.source:
        label, path = spec.split("=", 1)
        name, stage = label.split(":", 1)
        for r in map(json.loads, open(path)):
            if r["pattern_id"] != "rather_than":
                continue
            doc, st = r["doc_id"], stage
            if stage == "any":
                _, st, doc = doc.split("__", 2)
            if doc not in ids:
                continue
            if name == "original" and r["char_start"] >= body_len.get(doc, 0):
                continue
            item_id = f"{name}|{st}|{doc}|{r['char_start']}"
            if item_id in seen:
                continue
            seen.add(item_id)
            items.append({"item_id": item_id, "source": name, "stage": st, "doc_id": doc, "sentence": r["sentence"],
                          "local_start": r["local_start"], "local_end": r["local_end"],
                          "context_before": r.get("context_before", ""), "context_after": r.get("context_after", "")})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items_public.jsonl", "w") as f:
        f.writelines(json.dumps(i) + "\n" for i in items)
    print(len(items), "items")
    for key, n in sorted(Counter((i["source"], i["stage"]) for i in items).items()):
        print(f"  {key[0]:28s} {key[1]:10s} {n}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Stance ratings of the pilots (stance_llm.py output): per source and stage, the number of uses and of papers, the mean
gap X - Y, the mean rating of Y, the share of uses with Y < 0 and the share of inverted uses (X < Y). Also the number
of items with a valid X/Y pair and the estimated API cost of the extraction and rating, from the token usage recorded in
the outputs and the price table of paper_pipeline (an estimate, not a billing record).

Usage:
    python3 stance_pilot_summary.py ../data/stance_pilot [--model gpt-6-sol]
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper_pipeline" / "src"))
from paper_pipeline.utils import estimate_cost

ORDER = ["original", "tulu3-8b-sft", "tulu3-8b-dpo", "tulu3-8b", "claude-haiku-4-5-20251001", "claude-sonnet-4-5",
         "claude-sonnet-5-5", "claude-opus-5-5", "gpt-4o", "gpt-5.6-sol", "gpt-6-sol"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stance_dir")
    ap.add_argument("--model", default="gpt-6-sol")
    args = ap.parse_args()
    d = Path(args.stance_dir)
    items = {r["item_id"]: r for r in map(json.loads, open(d / "items_public.jsonl"))}
    xy = [json.loads(l) for l in open(d / "xy_alternatives.jsonl")]
    st = {r["item_id"]: r for r in map(json.loads, open(d / f"stance__{args.model}.jsonl"))}
    usage = lambda u: SimpleNamespace(prompt_tokens=u["prompt_tokens"], completion_tokens=u["completion_tokens"])
    cost = sum(estimate_cost(usage(r["usage"]), args.model) for r in xy + list(st.values()) if r.get("usage"))
    print(f"items {len(items)}; X/Y valid {sum(r['valid'] for r in xy)}/{len(xy)}; rated {len(st)}; estimated cost ${cost:.2f}\n")
    rows = collections.defaultdict(list)
    for i, r in st.items():
        it = items[i]
        rows[(it["source"], it["stage"])].append((r["x"], r["y"], it["doc_id"]))
    print(f"{'source':28s} {'stage':10s} {'uses':>5s} {'papers':>6s} {'gap':>6s} {'Y':>6s} {'Y<0':>5s} {'inv':>5s}")
    for src in ORDER:
        for stage in ("original", "generated", "revised"):
            v = rows.get((src, stage))
            if not v:
                continue
            n = len(v)
            print(f"{src:28s} {stage:10s} {n:5d} {len({doc for *_, doc in v}):6d} {statistics.mean(x - y for x, y, _ in v):+6.2f} "
                  f"{statistics.mean(y for _, y, _ in v):+6.2f} {100 * sum(y < 0 for _, y, _ in v) / n:4.0f}% "
                  f"{100 * sum(x < y for x, y, _ in v) / n:4.0f}%")


if __name__ == "__main__":
    main()

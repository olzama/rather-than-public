#!/usr/bin/env python3
r"""
Table of the open-model and Claude pilots: for each source, the number of papers, mean words, papers with
"rather than", its rate per 1,000 words in the first drafts and in the revised papers, and the rate of the
postposed "X, not Y" (the table "Other models" of the paper).

The pilot papers (data/generated/pilot.zip, unzipped) and the GPT papers of the same source papers
(data/generated/gpt_papers/batch2.zip, unzipped) are read through generation_view.combined, so a model folder may sit in
any of the generation directories. The papers are those listed in --ids-file (corpus<TAB>doc_id; the 20 pilot source
papers, data/generated/pilot_ids.tsv). Originals are <acl_dir>/<id>.txt and <arxiv_dir>/<id>.txt, body text; a source
without its original is skipped. Rates are per 1,000 words (\w+ tokens), averaged over papers, with the patterns of
antithesis_patterns.py.

Usage:
    python3 pilot_table.py ../data/generated/pilot ../data/generated/gpt_papers/batch2 \
        --ids-file ../data/generated/pilot_ids.tsv --acl-dir ../data/acl2019/text_body \
        --arxiv-dir ../data/arxiv2026/text_body \
        --models allenai_Llama-3.1-Tulu-3-8B-SFT allenai_Llama-3.1-Tulu-3-8B-DPO allenai_Llama-3.1-Tulu-3-8B \
                 claude-haiku-4-5-20251001 claude-sonnet-4-5 claude-sonnet-5-5 claude-opus-5-5 gpt-4o gpt-5.6-sol gpt-6-sol
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import compile_patterns
from generation_rates import measure
from generation_view import combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generation_dirs", nargs="+")
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--acl-dir", required=True)
    ap.add_argument("--arxiv-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    ids = [(c, i) for c, i in (l.split()[:2] for l in open(args.ids_file) if l.strip() and not l.startswith("#"))]
    g = combined(args.generation_dirs, args.models)
    compiled = compile_patterns()
    originals = [(Path(args.acl_dir if c == "acl2019" else args.arxiv_dir) / f"{i}.txt") for c, i in ids]
    rows = {"Original papers": ("-", originals)}
    for m in args.models:
        rows[m] = ("generated_papers", [g / m / "generated_papers" / f"{i}.md" for _, i in ids])
        rows[m + " revised"] = ("revised_papers", [g / m / "revised_papers" / f"{i}.md" for _, i in ids])
    stats = {}
    for name, (_, paths) in rows.items():
        paths = [p for p in paths if p.exists()]
        ms = [measure(p, compiled) for p in paths]
        stats[name] = ms
    print(f"{'source':45s} {'n':>3s} {'words':>7s} {'withRT':>6s} {'RT':>6s} {'RT rev.':>7s} {'X, not Y':>8s}")
    rate = lambda ms, k: statistics.mean(1000 * x[k] / x["words"] for x in ms)
    for name in ["Original papers"] + args.models:
        ms = stats[name]
        if not ms:
            continue
        rev = stats.get(name + " revised")
        rev_rate = f"{rate(rev, 'rather_than'):7.2f}" if rev else "      -"
        print(f"{name:45s} {len(ms):3d} {statistics.mean(x['words'] for x in ms):7.0f} {sum(x['rather_than'] > 0 for x in ms):6d} "
              f"{rate(ms, 'rather_than'):6.2f} {rev_rate} {rate(ms, 'x_not_y'):8.2f}")


if __name__ == "__main__":
    main()

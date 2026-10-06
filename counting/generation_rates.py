#!/usr/bin/env python3
r"""
Antithesis rates in dataset (3): LLM-written versions of ACL 2019 and
arXiv 2026 papers, compared with the original papers.

Each <generation_dir> holds one folder per model, each with generated_papers/
and revised_papers/ (one <paper_id>.md per paper); several generation runs
(batches) are read together, since a source paper is never drawn twice. Originals are
<acl_dir>/<paper_id>.txt for ACL 2019 ids and <arxiv_dir>/<paper_id>.txt
for arXiv ids (NNNN.NNNNN), body text. ACL 2019-based and arXiv-based
papers are reported separately. A paper is used only if it is present in
every source and none of its texts is flagged by language_filter.assess
(not usable English), or listed in --excluded.

Rates are per 1,000 words (\w+ tokens), averaged over papers, with the
same patterns as count_antithesis_patterns.py. Paired Wilcoxon
signed-rank tests on per-paper "rather than" rates compare each
generated set with the originals, and each revised set with its
generated set.

Usage:
    python3 generation_rates.py ../data/generated/gpt_papers/batch2 ../data/generated/gpt_papers/batch3 \
        --acl-dir ../../data/acl2019/text_body \
        [--arxiv-dir ../../data/arxiv2026/text_body] [--excluded ../data/excluded_documents.tsv] \
        [--models gpt-4o gpt-5.6-sol gpt-6-sol]
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import PATTERNS, compile_patterns
from language_filter import assess

WORD_RE = re.compile(r"\w+")
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
STAGES = ["generated_papers", "revised_papers"]


def measure(path, compiled):
    text = path.read_text(errors="replace")
    n = len(WORD_RE.findall(text))
    return {"words": n, "text": text, **{pid: len(rx.findall(text)) for pid, rx in compiled.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generation_dirs", nargs="+")
    ap.add_argument("--acl-dir", required=True)
    ap.add_argument("--arxiv-dir", default="../../data/arxiv2026/text_body")
    ap.add_argument("--excluded")
    ap.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-5.6-sol", "gpt-6-sol"])
    args = ap.parse_args()
    from generation_view import combined
    g = combined(args.generation_dirs, args.models)  # all runs as one folder
    compiled = compile_patterns()
    originals = {"ACL 2019": Path(args.acl_dir), "arXiv 2026": Path(args.arxiv_dir)}
    excluded = set()
    if args.excluded:
        for line in list(open(args.excluded))[1:]:
            corpus, doc_id = line.split("\t")[:2]
            if corpus in ("acl2019", "arxiv2026"):
                excluded.add(doc_id)

    gen_sources = {f"{m} {s.split('_')[0]}": g / m / s for m in args.models for s in STAGES}
    common = set.intersection(*({p.stem for p in d.glob("*.md")} for d in gen_sources.values()))
    for group, orig_dir in originals.items():
        ids = {i for i in common if bool(ARXIV_ID_RE.match(i)) == (group == "arXiv 2026")
               and (orig_dir / f"{i}.txt").exists()}
        if not ids:
            continue
        sources = {f"{group} original": (orig_dir, ".txt"), **{k: (d, ".md") for k, d in gen_sources.items()}}
        report(group, sources, ids, excluded, compiled, args.models)


def report(group, sources, ids, excluded, compiled, models):
    data, dropped = {}, {}
    for name, (d, ext) in sources.items():
        data[name] = {i: measure(d / f"{i}{ext}", compiled) for i in sorted(ids)}
    for i in sorted(ids):
        if i in excluded:
            dropped[i] = "excluded list"
            continue
        for name in sources:
            reason = assess(data[name][i]["text"], 30, 200, 0.5)[0]
            if reason:
                dropped[i] = f"{name}: {reason}"
                break
    keep = sorted(ids - set(dropped))
    print(f"\n=== {group}: papers in every source: {len(ids)}; dropped {len(dropped)}; used {len(keep)}")
    for i, why in sorted(dropped.items()):
        print(f"  dropped {i} ({why})")
    if len(keep) < 2:
        return
    orig = f"{group} original"

    def rate(name, pid):
        return [1000 * data[name][i][pid] / data[name][i]["words"] for i in keep]

    print(f"\n{'source':22s} {'words':>6s} {'median':>6s} {'papers RT':>9s}" + "".join(f"{lab[:14]:>15s}" for _, _, lab, _, _ in PATTERNS))
    for name in sources:
        words = statistics.mean(data[name][i]["words"] for i in keep)
        median = statistics.median(data[name][i]["words"] for i in keep)
        with_rt = sum(data[name][i]["rather_than"] > 0 for i in keep)
        print(f"{name:22s} {words:6.0f} {median:6.0f} {with_rt:9d}" + "".join(f"{statistics.mean(rate(name, pid)):15.3f}" for pid, *_ in PATTERNS))

    print("\npaired Wilcoxon, per-paper 'rather than' rate:")
    base = rate(orig, "rather_than")
    for m in models:
        gen, rev = rate(f"{m} generated", "rather_than"), rate(f"{m} revised", "rather_than")
        print(f"  {m}: generated vs original p = {stats.wilcoxon(gen, base).pvalue:.3g}; "
              f"revised vs generated p = {stats.wilcoxon(rev, gen).pvalue:.3g} "
              f"(higher in revised for {sum(r > x for r, x in zip(rev, gen))}/{len(keep)} papers)")

if __name__ == "__main__":
    main()

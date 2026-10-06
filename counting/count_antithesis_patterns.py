#!/usr/bin/env python3
r"""
Count occurrences of a set of antithesis-family constructions (not just
"rather than") across a text corpus, one pass per file over all patterns.
See ../data_collection/antithesis_patterns.py for the exact list and the
tier taxonomy behind it. For per-instance extraction (matched text,
sentence, document id, character span), see
../data_collection/extract_antithesis_instances.py -- same pattern list,
imported from the same place.

These are regex heuristics over raw text, not a parse -- several are
noisy by construction (documented per-pattern below). Treat this as a
first pass, the same way "rather than" itself was treated: informative
at corpus scale, not a claim about any individual sentence.

Usage:
    python3 count_antithesis_patterns.py <text_dir> [--glob "*.txt"] [--json out.json] \
        [--exclude ../data/excluded_documents.tsv --exclude-corpus acl2019]

--exclude skips documents listed in language_filter.py's output for
--exclude-corpus (matched on the file stem).

--compare OTHER_DIR [--compare-corpus NAME] also counts a second corpus and
tests, per pattern, whether per-file rates differ (two-sided Mann-Whitney):
    python3 count_antithesis_patterns.py ../data/acl2019/text_body \
        --exclude ../data/excluded_documents.tsv --exclude-corpus acl2019 \
        --compare ../data/arxiv2026/text_body --compare-corpus arxiv2026

Prints a table: pattern x (occurrences, files-with-hit%, mean/median per
1000 words). Matches per pattern are counted independently (a passage
matching Tier 3's "not just X but Y" is NOT also counted under Tier 2's
generic "not X but Y", since the Tier 2 pattern explicitly excludes
not-just/only/merely -- see each regex's comment).
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from antithesis_patterns import PATTERNS

WORD_RE = re.compile(r"\w+")


def count_in_file(path, compiled):
    text = path.read_text(errors="replace")
    n_words = len(WORD_RE.findall(text))
    counts = {pid: len(rx.findall(text)) for pid, rx in compiled.items()}
    return n_words, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text_dir")
    p.add_argument("--glob", default="*.txt")
    p.add_argument("--json", default=None)
    p.add_argument("--exclude", help="TSV from language_filter.py")
    p.add_argument("--exclude-corpus", help="corpus name in the TSV to apply")
    p.add_argument("--compare", help="second text directory: test per-file rates against it")
    p.add_argument("--compare-corpus", help="corpus name in the --exclude TSV for --compare")
    args = p.parse_args()

    compiled = {pid: re.compile(pattern, re.I) for pid, _tier, _label, pattern, _note in PATTERNS}
    text_dir = Path(args.text_dir)
    files = sorted(text_dir.glob(args.glob))
    if not files:
        sys.exit(f"no files matching {args.glob} under {text_dir}")
    if args.exclude:
        skip = {l.split("\t")[1] for l in list(open(args.exclude))[1:] if l.split("\t")[0] == args.exclude_corpus}
        n_before = len(files)
        files = [f for f in files if f.stem not in skip]
        print(f"excluded {n_before - len(files)} documents listed for {args.exclude_corpus}")

    per_file = []
    for f in files:
        n_words, counts = count_in_file(f, compiled)
        per_file.append({"file": f.name, "words": n_words, "counts": counts})

    total_files = len(per_file)
    print(f"corpus: {text_dir}  files: {total_files}")
    print(f"{'tier':<5}{'pattern':<28}{'total':>8}{'files%':>9}{'mean/1k':>10}{'median/1k':>11}")

    summary = {}
    for pid, tier, label, _pattern, note in PATTERNS:
        total = sum(r["counts"][pid] for r in per_file)
        files_with = sum(1 for r in per_file if r["counts"][pid] > 0)
        rates = [1000 * r["counts"][pid] / r["words"] for r in per_file if r["words"] > 0]
        mean_rate = statistics.mean(rates)
        median_rate = statistics.median(rates)
        print(f"{tier:<5}{label:<28}{total:>8}{100*files_with/total_files:>8.1f}%{mean_rate:>10.4f}{median_rate:>11.4f}")
        summary[pid] = {
            "tier": tier, "label": label, "total": total,
            "files_with_hit": files_with, "files_with_hit_pct": 100 * files_with / total_files,
            "mean_per_1000w": mean_rate, "median_per_1000w": median_rate, "note": note,
        }

    if args.json:
        with open(args.json, "w") as out:
            json.dump({"corpus": str(text_dir), "total_files": total_files, "patterns": summary}, out, indent=2)

    if args.compare:
        from scipy.stats import mannwhitneyu
        other = sorted(Path(args.compare).glob(args.glob))
        if args.exclude:
            skip = {l.split("\t")[1] for l in list(open(args.exclude))[1:] if l.split("\t")[0] == args.compare_corpus}
            other = [f for f in other if f.stem not in skip]
        other = [count_in_file(f, compiled) for f in other]
        print(f"\ncompare: {args.compare}  files: {len(other)}  (two-sided Mann-Whitney on per-file rates)")
        print(f"{'pattern':<28}{'mean/1k':>10}{'other':>10}{'p':>12}")
        for pid, _tier, label, _pattern, _note in PATTERNS:
            a = [1000 * r["counts"][pid] / r["words"] for r in per_file if r["words"] > 0]
            b = [1000 * c[pid] / w for w, c in other if w > 0]
            p_val = mannwhitneyu(a, b, alternative="two-sided").pvalue
            print(f"{label:<28}{statistics.mean(a):>10.4f}{statistics.mean(b):>10.4f}{p_val:>12.2g}")


if __name__ == "__main__":
    main()

"""Draw a seeded random sample of papers for generation, without generating.

Uses the same selection as `main --num-papers` (config `sources`, split
evenly, excluded and invalid records skipped) and writes one
"corpus<TAB>doc_id" line per paper, for --ids-file.

Usage (from the repository root):
    PYTHONPATH=paper_pipeline/src python3 -m paper_pipeline.sample_ids \
        --n 100 --seed 20260928 --out data/generation/sample_ids.tsv \
        [--sources acl2019 arxiv2026] [--batch NAME] [--exclude-ids FILE ...] \
        [--config paper_pipeline/config/config.yaml]
"""
import argparse
import logging
from pathlib import Path

from .config_loader import load_config
from .main import DEFAULT_CONFIG
from .seeds import draw_batch, load_source_ids, read_registry, select_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--exclude-ids", type=Path, nargs="+",
                        help="Further files of papers never to draw ('corpus<TAB>doc_id')")
    parser.add_argument("--batch", help="Record the draw in the used-papers registry under this batch name "
                                        "(as main --batch does); without it the draw is not recorded")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    config = load_config(args.config)
    sources = {k: Path(v) for k, v in config.sources.items() if args.sources is None or k in args.sources}
    excluded = Path(config.excluded_documents) if config.excluded_documents else None
    exclude_files = args.exclude_ids or []
    extra = [pair for path in exclude_files for pair in load_source_ids(path)]
    registry = Path(config.used_papers)
    if args.batch:
        selected, _ = draw_batch(sources, excluded, registry, args.batch, args.n, args.seed, extra_used=extra)
    else:
        used = [(c, d) for _, c, d in read_registry(registry)] + extra
        selected = select_seeds(sources, excluded, args.n, args.seed, used=used)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(f"# {args.n} papers from {sorted(sources)}; seed {args.seed}; excluded {excluded}; "
                  f"registry {registry}; batch {args.batch}; exclude_ids {[str(p) for p in exclude_files]}\n")
        out.writelines(f"{corpus}\t{seed['doc_id']}\n" for corpus, seed in selected)
    counts = {c: sum(1 for x, _ in selected if x == c) for c in sources}
    print(f"wrote {len(selected)} papers ({counts}) to {args.out}")


if __name__ == "__main__":
    main()

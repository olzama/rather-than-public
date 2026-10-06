import argparse
import logging
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

from .config_loader import load_config
from .pipeline import PaperPipeline
from .seeds import draw_batch, filename_stem, iter_seeds, load_excluded, load_ids, load_source_ids, select_seeds
from tqdm import tqdm

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
ANTITHESIS_SCRIPT = REPOSITORY_ROOT / "data_collection" / "extract_antithesis_instances.py"
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, review, and revise ACL-style academic papers."
    )
    parser.add_argument(
        "--stage", choices=("all", "generate", "review", "revise", "extract"),
        default="all", help="Stage to run independently (default: all)",
    )
    parser.add_argument("--review", type=Path, help="Existing review text for --stage revise")
    parser.add_argument("--reviews-dir", type=Path, help="Matching reviews for folder revision")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing review/revision outputs")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--papers-dir", type=Path, help="Paper folder for review, revise, or extract")
    source.add_argument("--input-jsonl", type=Path, help="Override config input_jsonl")
    parser.add_argument("--num-papers", type=int,
                        help="Generate N papers drawn at random, split evenly across --sources "
                             "(with --input-jsonl: the first N valid records). Default: all")
    parser.add_argument("--sources", nargs="+",
                        help="Config sources to draw from (default: all configured, e.g. acl2019 arxiv2026)")
    parser.add_argument("--seed", type=int, default=20260928, help="Random seed for --num-papers")
    parser.add_argument("--ids-file", type=Path,
                        help="Generate exactly the papers listed in FILE: 'corpus<TAB>doc_id' per line "
                             "(a bare doc_id means acl2019), e.g. a sample_ids.tsv from an earlier run")
    parser.add_argument("--batch",
                        help="Name of the generation batch (required when drawing papers). The first run "
                             "of a batch draws papers not in the used-papers registry and appends them to it; "
                             "later runs with the same name (e.g. other models) reuse exactly those papers")
    parser.add_argument("--exclude-ids", type=Path, nargs="+",
                        help="Further files of papers never to draw ('corpus<TAB>doc_id' per line), "
                             "in addition to the used-papers registry")
    parser.add_argument("--excluded", type=Path,
                        help="Override config excluded_documents (TSV from language_filter.py)")
    source.add_argument(
        "--paper",
        "-p",
        type=Path,
        default=None,
        help="Existing UTF-8 plain-text paper to review and revise, skipping generation",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--generated-dir",
        type=str,
        default=None,
        help="Output override (default: <model>/generated_papers)",
    )
    parser.add_argument(
        "--evaluated-dir",
        type=str,
        default=None,
        help="Output override (default: <model>/evaluated_papers)",
    )
    parser.add_argument(
        "--revised-dir",
        type=str,
        default=None,
        help="Output override (default: <model>/revised_papers)",
    )
    args = parser.parse_args()
    if args.num_papers is not None and args.num_papers < 1:
        parser.error("--num-papers must be positive")
    if (args.input_jsonl is not None or args.num_papers is not None or args.sources is not None
            or args.ids_file is not None or args.excluded is not None or args.exclude_ids is not None
            or args.batch is not None) and (
        args.stage not in ("all", "generate") or args.paper is not None
    ):
        parser.error("--input-jsonl, --num-papers, --sources, --ids-file, --batch, --exclude-ids and "
                     "--excluded require dataset generation")
    if args.batch is not None and (args.ids_file is not None or args.input_jsonl is not None):
        parser.error("--batch draws from the configured sources; it cannot be combined with --ids-file or --input-jsonl")
    if args.exclude_ids is not None and args.input_jsonl is not None:
        parser.error("--exclude-ids requires --sources/--num-papers sampling (not --input-jsonl)")
    if args.input_jsonl is not None and args.sources is not None:
        parser.error("--sources cannot be combined with --input-jsonl")
    if args.stage == "generate" and args.paper is not None:
        parser.error("--stage generate does not accept --paper")
    if args.stage in ("review", "revise") and args.paper is None and args.papers_dir is None:
        parser.error(f"--stage {args.stage} requires --paper FILE or --papers-dir DIR")
    if args.stage == "revise":
        if args.paper is not None and (args.review is None or args.reviews_dir is not None):
            parser.error("Single-paper revision requires --review FILE, without --reviews-dir")
        if args.papers_dir is not None and (args.reviews_dir is None or args.review is not None):
            parser.error("Folder revision requires --reviews-dir DIR, without --review")
    if args.stage == "extract" and args.papers_dir is None:
        parser.error("--stage extract requires --papers-dir DIR")
    if args.review is not None and args.stage != "revise":
        parser.error("--review is only used with --stage revise")
    if args.reviews_dir is not None and args.stage != "revise":
        parser.error("--reviews-dir is only used with --stage revise")
    if args.papers_dir is not None and args.stage not in ("review", "revise", "extract"):
        parser.error("--papers-dir is only used with review, revise, or extract")
    if args.skip_existing and args.stage not in ("review", "revise"):
        parser.error("--skip-existing is only used with review or revise")
    return args


def configure_output_dirs(args: argparse.Namespace, model_name: str) -> Path:
    """Group default outputs under a filename-safe model name; retain overrides."""
    model_folder = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip(".")
    if not model_folder:
        raise ValueError("Model name must contain a valid directory name")
    root = Path(model_folder)
    for attribute, folder in (
        ("generated_dir", "generated_papers"),
        ("evaluated_dir", "evaluated_papers"),
        ("revised_dir", "revised_papers"),
    ):
        if getattr(args, attribute) is None:
            setattr(args, attribute, root / folder)
    return root


@contextmanager
def model_logging(model_dir: Path):
    """Append run logs to the model folder and release the handler on exit."""
    model_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(model_dir / "pipeline.log", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    try:
        logger.info("Default output root: %s", model_dir.resolve())
        yield
    except Exception:
        logger.exception("Pipeline failed")
        raise
    finally:
        root_logger.removeHandler(handler)
        handler.close()
        root_logger.setLevel(previous_level)


def write_text(directory: str, filename: str, text: str) -> Path:
    path = Path(directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    logger.info("Wrote %s", path.resolve())
    return path


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Input is empty: {path}")
    return text


def launch_script(script_path, *arguments):
    result = subprocess.run(
        [sys.executable, script_path, *map(str, arguments)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def extract_antithesis(papers_dir: str) -> None:
    """Write antithesis matches beside the papers in the supplied directory."""
    papers_path = Path(papers_dir).resolve()
    antithesis_path = papers_path / "antithesis" / "antithesis_instances.jsonl"
    antithesis_path.parent.mkdir(parents=True, exist_ok=True)
    launch_script(ANTITHESIS_SCRIPT, papers_path, antithesis_path, "--glob", "*.md")


def track_antithesis(args: argparse.Namespace) -> None:
    """Refresh matches for saved generation/revision outputs, even after failures."""
    directories = []
    if args.stage == "generate" or (args.stage == "all" and args.paper is None):
        directories.append(args.generated_dir)
    if args.stage in ("all", "revise"):
        directories.append(args.revised_dir)
    for directory in directories:
        if any(Path(directory).glob("*.md")):
            logger.info("Extracting antithesis constructions from %s", directory)
            extract_antithesis(directory)


def review_paper(args: argparse.Namespace, pipeline: PaperPipeline,
                 stem: str, paper_text: str) -> tuple[str, float]:
    logger.info("Reviewing: %s", stem)
    review, cost = pipeline.evaluator.evaluate_paper(paper_text)
    write_text(args.evaluated_dir, f"{stem}.md", review)
    return review, cost


def revise_paper(args: argparse.Namespace, pipeline: PaperPipeline,
                 stem: str, paper_text: str, review_text: str) -> float:
    logger.info("Revising: %s", stem)
    revised, cost = pipeline.reviser.revise_paper(paper_text, review_text)
    write_text(args.revised_dir, f"{stem}.md", revised)
    return cost


def generate_papers(args: argparse.Namespace, pipeline: PaperPipeline,
                    input_jsonl: Path, full_pipeline: bool = False) -> float:
    total_cost = 0.0
    succeeded = failed = 0
    selected = getattr(args, "selected_seeds", None)
    if selected is not None:
        seeds, total = iter(selected), len(selected)
    else:
        seeds = iter_seeds(input_jsonl, args.num_papers,
                           exclude=getattr(args, "exclude_ids", frozenset()),
                           only=getattr(args, "only_ids", None))
        total = args.num_papers
    for seed in tqdm(
        seeds,
        total=total,
        desc="Generating papers",
        unit="paper",
    ):
        doc_id = seed["doc_id"]
        stem = filename_stem(doc_id)
        logger.info("Generating doc_id=%s", doc_id)
        try:
            paper, cost = pipeline.generator.generate_paper(seed)
            total_cost += cost
            write_text(args.generated_dir, f"{stem}.md", paper)
            if full_pipeline:
                review, cost = review_paper(args, pipeline, stem, paper)
                total_cost += cost
                total_cost += revise_paper(args, pipeline, stem, paper, review)
            succeeded += 1
        except Exception:
            failed += 1
            logger.exception("Failed doc_id=%s", doc_id)
    logger.info("Paper summary: %d succeeded, %d failed; estimated cost $%.4f",
                succeeded, failed, total_cost)
    if failed:
        raise SystemExit(1)
    return total_cost


def run_all(args: argparse.Namespace, pipeline: PaperPipeline) -> float:
    if args.paper is not None:
        paper = read_text(args.paper)
        review, cost = review_paper(args, pipeline, args.paper.stem, paper)
        return cost + revise_paper(args, pipeline, args.paper.stem, paper, review)
    return generate_papers(args, pipeline, args.input_jsonl, full_pipeline=True)


def prepare_paper_jobs(args: argparse.Namespace) -> tuple[list, int]:
    """Validate all pending inputs before loading a model or processing the batch."""
    if args.papers_dir is not None:
        if not args.papers_dir.is_dir():
            raise ValueError(f"Paper directory does not exist: {args.papers_dir}")
        papers = sorted(path for path in args.papers_dir.iterdir() if path.is_file() and path.suffix in (".txt", ".md"))
        if not papers:
            raise ValueError(f"No .txt or .md papers found in {args.papers_dir}")
    else:
        papers = [args.paper]
    jobs = []
    skipped = 0
    errors = []
    stems = set()
    for paper in papers:
        if paper.stem.casefold() in stems:
            errors.append(f"Duplicate output filename for paper: {paper}")
            continue
        stems.add(paper.stem.casefold())
        directory = args.evaluated_dir if args.stage == "review" else args.revised_dir
        output = Path(directory) / f"{paper.stem}.md"
        if args.skip_existing and output.is_file():
            logger.info("Skipping existing output: %s", output)
            skipped += 1
            continue
        try:
            paper_text = read_text(paper)
            review_text = None
            if args.stage == "revise":
                review = (args.reviews_dir / f"{paper.stem}.md"
                          if args.papers_dir is not None else args.review)
                review_text = read_text(review)
            jobs.append((paper, paper_text, review_text))
        except (OSError, ValueError) as error:
            errors.append(f"{paper}: {error}")
    if errors:
        raise ValueError("Cannot start batch:\n" + "\n".join(errors))
    return jobs, skipped


def run_paper_jobs(args: argparse.Namespace, pipeline: PaperPipeline,
                   jobs: list, skipped: int) -> float:
    cost = 0.0
    failed = 0
    succeeded = 0
    for paper, paper_text, review_text in jobs:
        try:
            if args.stage == "review":
                _, item_cost = review_paper(args, pipeline, paper.stem, paper_text)
            else:
                item_cost = revise_paper(args, pipeline, paper.stem, paper_text, review_text)
            cost += item_cost
            succeeded += 1
        except Exception:
            failed += 1
            logger.exception("Failed to %s %s", args.stage, paper)
    logger.info("%s summary: %d succeeded, %d skipped, %d failed; successful outputs cost $%.4f",
                args.stage, succeeded, skipped, failed, cost)
    if failed:
        raise SystemExit(1)
    return cost


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.stage == "extract":
        extract_antithesis(args.papers_dir)
        return

    config = load_config(args.config)
    model_dir = configure_output_dirs(args, config.model.name)
    with model_logging(model_dir):
        for directory in (args.generated_dir, args.evaluated_dir, args.revised_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)
        run_pipeline(args, config)


def run_pipeline(args: argparse.Namespace, config) -> None:
    """Run the configured model stages with output paths and logging ready."""
    if args.stage == "generate" or (args.stage == "all" and args.paper is None):
        excluded = args.excluded or (Path(config.excluded_documents) if config.excluded_documents else None)
        if excluded is not None and not excluded.is_file():
            sys.exit(f"excluded_documents file not found: {excluded} (set excluded_documents: null to disable)")
        if excluded is None:
            logger.warning("No excluded_documents configured: excluded papers may be generated")
        if args.input_jsonl is not None:
            # one explicit file, read in order (ACL 2019 exclusions apply)
            if excluded is not None:
                args.exclude_ids = load_excluded(excluded)
            if args.ids_file is not None:
                args.only_ids = [doc_id for _, doc_id in load_source_ids(args.ids_file)]
        else:
            sources = {name: Path(path) for name, path in config.sources.items()
                       if args.sources is None or name in args.sources}
            unknown = set(args.sources or ()) - set(config.sources)
            if unknown or not sources:
                sys.exit(f"unknown or no sources: {sorted(unknown)}; configured: {sorted(config.sources)}")
            extra = [pair for path in args.exclude_ids or () for pair in load_source_ids(path)]
            if args.ids_file is not None:
                selected = select_seeds(sources, excluded, only=load_source_ids(args.ids_file))
                origin = f"ids_file={args.ids_file}"
            else:
                if args.batch is None:
                    sys.exit("--batch NAME is required when drawing papers: the papers drawn are appended to "
                             "the used-papers registry so later batches never repeat them")
                registry = Path(getattr(config, "used_papers", "paper_pipeline/used_papers.tsv"))
                try:
                    selected, new = draw_batch(sources, excluded, registry, args.batch, args.num_papers,
                                               args.seed, extra_used=extra)
                except ValueError as err:
                    sys.exit(str(err))
                origin = f"batch={args.batch}, seed={args.seed}, registry={registry}"
                if new:
                    logger.info("Batch %s: drew %d new papers and appended them to %s -- commit that file "
                                "so later batches skip them", args.batch, len(selected), registry)
                else:
                    logger.info("Batch %s: reusing its %d papers from %s", args.batch, len(selected), registry)
            args.selected_seeds = [seed for _, seed in selected]
            record = Path(args.generated_dir).parent / "sample_ids.tsv"
            record.parent.mkdir(parents=True, exist_ok=True)
            with open(record, "w", encoding="utf-8") as out:
                out.write(f"# {len(selected)} papers from {sorted(sources)}; {origin}; excluded={excluded}; "
                          f"exclude_ids={[str(p) for p in args.exclude_ids or ()]}\n")
                out.writelines(f"{corpus}\t{seed['doc_id']}\n" for corpus, seed in selected)
            logger.info("Selected %d papers (%s); list written to %s", len(selected),
                        ", ".join(f"{c}: {sum(1 for x, _ in selected if x == c)}" for c in sources), record)
    if args.input_jsonl is None:
        args.input_jsonl = Path(config.input_jsonl)

    jobs, skipped = ([], 0)
    if args.stage in ("review", "revise"):
        jobs, skipped = prepare_paper_jobs(args)
        if not jobs:
            logger.info("%s summary: 0 succeeded, %d skipped, 0 failed", args.stage, skipped)
            track_antithesis(args)
            return

    load_dotenv()
    pipeline = PaperPipeline(config)
    try:
        if args.stage == "generate":
            cost = generate_papers(args, pipeline, args.input_jsonl)
        elif args.stage in ("review", "revise"):
            cost = run_paper_jobs(args, pipeline, jobs, skipped)
        else:
            cost = run_all(args, pipeline)
    finally:
        track_antithesis(args)
    if cost:
        logger.info("Estimated total cost: $%.4f", cost)


if __name__ == "__main__":
    main()

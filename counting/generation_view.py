"""
One view of several generation runs (batches of GPT papers), for the scripts
that read GPT papers. Each run folder holds <model>/{generated,evaluated,
revised}_papers/<paper_id>.md; a source paper is never drawn twice, so the runs
can be combined. The view is a temporary folder of symbolic links with the same
layout, plus <model>/sample_ids.tsv (corpus, doc_id), taken from each run's own
sample_ids.tsv or, if a run has none, from the paper registry
(paper_pipeline/used_papers.tsv) rows whose batch name is the run folder's name.
"""
import sys
import tempfile
from pathlib import Path

STAGES = ["generated_papers", "evaluated_papers", "revised_papers"]
REGISTRY = Path(__file__).resolve().parent.parent / "paper_pipeline" / "used_papers.tsv"


def registry_ids(batch):
    rows = [l.rstrip("\n").split("\t") for l in open(REGISTRY) if l.strip() and not l.startswith("#")]
    return [(corpus, doc) for b, corpus, doc in rows if b == batch]


def combined(dirs, models):
    view = Path(tempfile.mkdtemp(prefix="generation_"))
    ids = {m: [] for m in models}
    for gd in map(Path, dirs):
        for m in models:
            for stage in STAGES:
                (view / m / stage).mkdir(parents=True, exist_ok=True)
                for f in (gd / m / stage).glob("*.md"):
                    link = view / m / stage / f.name
                    if link.exists():
                        sys.exit(f"{f.name} occurs in more than one generation run")
                    link.symlink_to(f.resolve())
            own = gd / m / "sample_ids.tsv"
            if own.exists():
                ids[m] += [tuple(l.rstrip("\n").split("\t")[:2]) for l in open(own) if l.strip() and not l.startswith("#")]
            else:
                ids[m] += registry_ids(gd.name)
    for m in models:
        with open(view / m / "sample_ids.tsv", "w") as f:
            f.writelines(f"{c}\t{d}\n" for c, d in ids[m])
    return view

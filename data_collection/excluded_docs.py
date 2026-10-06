r"""
Documents excluded from every sample: the list written by
language_filter.py (non-English or near-empty text), at
../data/excluded_documents.tsv. The sampling scripts filter instances
through drop_excluded() right after loading them, so no pool item can be
drawn from an excluded document. A missing list is an error.

Instance corpora map to the list's corpus names: acl2019 -> acl2019;
arxiv2026 and high_count_arxiv2026 -> arxiv2026; arxiv_v1 and
arxiv_latest -> arxiv_versions (doc_ids there carry the _vN suffix).
"""
import sys
from pathlib import Path

DEFAULT_TSV = Path(__file__).resolve().parent.parent / "data" / "excluded_documents.tsv"
LIST_CORPUS = {"acl2019": "acl2019", "arxiv2026": "arxiv2026", "high_count_arxiv2026": "arxiv2026",
               "arxiv_v1": "arxiv_versions", "arxiv_latest": "arxiv_versions"}


def load(tsv=DEFAULT_TSV):
    """Set of (list corpus, doc_id)."""
    tsv = Path(tsv)
    if not tsv.is_file():
        sys.exit(f"excluded-documents list not found: {tsv} (run language_filter.py)")
    rows = [line.rstrip("\n").split("\t") for line in open(tsv, encoding="utf-8")][1:]
    return {(r[0], r[1]) for r in rows if len(r) > 1}


def drop_excluded(records, tsv=DEFAULT_TSV, corpus=None):
    """Records (dicts with doc_id and, unless `corpus` is given, corpus) not from excluded documents."""
    excluded = load(tsv)
    kept = [r for r in records
            if (LIST_CORPUS.get(corpus or r["corpus"], corpus or r["corpus"]), r["doc_id"]) not in excluded]
    if len(kept) < len(records):
        print(f"excluded {len(records) - len(kept)} instances from documents in {tsv}", file=sys.stderr)
    return kept

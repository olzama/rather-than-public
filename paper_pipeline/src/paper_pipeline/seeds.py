"""Stream validated title/abstract seeds without retaining source paper bodies."""
import gzip
import json
import logging
import random
import re
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

logger = logging.getLogger(__name__)


def filename_stem(doc_id: str) -> str:
    """Reversible percent encoding; ordinary ACL identifiers remain unchanged."""
    stem = quote(doc_id, safe="-_.", encoding="utf-8")
    # Avoid hidden/traversal names, trailing dots, and Windows device names.
    if stem.startswith('.'):
        stem = '%2E' + stem[1:]
    if stem.endswith('.'):
        stem = stem[:-1] + '%2E'
    if re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', stem):
        stem = f'%{ord(stem[0]):02X}' + stem[1:]
    if len(stem.encode('utf-8')) > 240:
        raise ValueError('doc_id is too long for a safe output filename')
    return stem


def load_excluded(path: Path, corpus: str = 'acl2019') -> frozenset[str]:
    """doc_ids listed for `corpus` in language_filter.py's TSV (header row first)."""
    with open(path, encoding='utf-8') as source:
        rows = [line.rstrip('\n').split('\t') for line in source][1:]
    return frozenset(row[1] for row in rows if len(row) > 1 and row[0] == corpus)


def load_ids(path: Path) -> list[str]:
    """One doc_id per line; blank lines and lines starting with '#' are ignored."""
    with open(path, encoding='utf-8') as source:
        return [line.strip() for line in source if line.strip() and not line.startswith('#')]


def load_source_ids(path: Path, default_corpus: str = 'acl2019') -> list[tuple[str, str]]:
    """(corpus, doc_id) pairs from lines "corpus<TAB>doc_id"; a bare doc_id means default_corpus."""
    pairs = []
    for line in load_ids(path):
        corpus, _, doc_id = line.partition('\t')
        pairs.append((corpus, doc_id) if doc_id else (default_corpus, corpus))
    return pairs


def select_seeds(sources: dict[str, Path], excluded: Path | None, num: int | None = None,
                 seed: int = 0, only: list[tuple[str, str]] | None = None,
                 used: list[tuple[str, str]] | None = None) -> list[tuple[str, dict[str, str]]]:
    """(corpus, seed) pairs from several sections files.

    Excluded doc_ids (language_filter.py's TSV, per corpus) and invalid
    records are never selected, nor are the (corpus, doc_id) pairs in `used`
    (papers generated in earlier runs). With `only`, exactly those pairs, in that
    order (asking for an excluded or missing one is an error). With `num`, a
    random sample of `num` records, split evenly across sources (earlier
    sources take the remainder) and shuffled, reproducible from `seed`.
    Otherwise every valid record.
    """
    pools = {}
    for corpus, path in sources.items():
        exclude = load_excluded(excluded, corpus) if excluded else frozenset()
        exclude |= {doc_id for c, doc_id in used or () if c == corpus}
        pools[corpus] = {s['doc_id']: s for s in iter_seeds(Path(path), exclude=exclude)}
    if only is not None:
        missing = [pair for pair in only if pair[0] not in pools or pair[1] not in pools[pair[0]]]
        if missing:
            raise ValueError(f'requested doc_ids not available (excluded, invalid, or absent): {missing[:10]}')
        return [(corpus, pools[corpus][doc_id]) for corpus, doc_id in only]
    if num is None:
        return [(corpus, s) for corpus, pool in pools.items() for s in pool.values()]
    rng = random.Random(seed)
    corpora = list(pools)
    picked = []
    for index, corpus in enumerate(corpora):
        k = num // len(corpora) + (1 if index < num % len(corpora) else 0)
        if k > len(pools[corpus]):
            raise ValueError(f'{corpus}: {k} requested, {len(pools[corpus])} eligible')
        picked += [(corpus, pools[corpus][doc_id]) for doc_id in rng.sample(sorted(pools[corpus]), k)]
    rng.shuffle(picked)
    return picked


BATCH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def read_registry(path: Path) -> list[tuple[str, str, str]]:
    """(batch, corpus, doc_id) rows of the used-papers registry; missing file = none."""
    if not Path(path).is_file():
        return []
    rows = []
    with open(path, encoding='utf-8') as source:
        for line in source:
            if line.strip() and not line.startswith('#'):
                batch, corpus, doc_id = line.rstrip('\n').split('\t')
                rows.append((batch, corpus, doc_id))
    return rows


def draw_batch(sources: dict[str, Path], excluded: Path | None, registry: Path, batch: str,
               num: int | None = None, seed: int = 0,
               extra_used: list[tuple[str, str]] | None = None) -> tuple[list[tuple[str, dict[str, str]]], bool]:
    """Papers of generation batch `batch`, recorded in one registry of all papers ever drawn.

    If the registry already has `batch`, its papers are returned again (so every
    model of a batch writes from the same papers); asking for a different
    number is an error. Otherwise `num` papers are drawn at random (as in
    select_seeds) from those not in the registry nor in `extra_used`, and are
    appended to the registry under `batch` before being returned. Returns
    (selected, newly_drawn).
    """
    if not BATCH_NAME_RE.match(batch):
        raise ValueError(f'invalid batch name {batch!r}: use letters, digits, ".", "_", "-"')
    rows = read_registry(registry)
    mine = [(corpus, doc_id) for b, corpus, doc_id in rows if b == batch]
    if mine:
        if num is not None and num != len(mine):
            raise ValueError(f'batch {batch!r} already has {len(mine)} papers in {registry}; '
                             f'--num-papers {num} does not match (use a new batch name for a new draw)')
        return select_seeds(sources, excluded, only=mine), False
    used = [(corpus, doc_id) for _, corpus, doc_id in rows] + list(extra_used or ())
    selected = select_seeds(sources, excluded, num, seed, used=used)
    Path(registry).parent.mkdir(parents=True, exist_ok=True)
    new_file = not Path(registry).is_file()
    with open(registry, 'a', encoding='utf-8') as out:
        if new_file:
            out.write('# batch\tcorpus\tdoc_id: every paper drawn for generation; draws never repeat these\n')
        out.writelines(f'{batch}\t{corpus}\t{s["doc_id"]}\n' for corpus, s in selected)
    return selected, True


def iter_seeds(path: Path, limit: int | None = None, exclude: frozenset[str] = frozenset(),
               only: list[str] | None = None) -> Iterator[dict[str, str]]:
    """Yield valid unique records in input order; report and skip bad lines.

    Records whose doc_id is in `exclude` are skipped. With `only`, just those
    doc_ids are yielded; asking for an excluded doc_id is an error.
    """
    if limit is not None and limit < 1:
        raise ValueError('Paper limit must be positive')
    wanted = set(only) if only is not None else None
    if wanted is not None and wanted & exclude:
        raise ValueError(f'excluded doc_ids requested: {sorted(wanted & exclude)}')
    found: set[str] = set()
    n_excluded = 0
    seen: set[str] = set()
    filenames: set[str] = set()
    count = 0
    opener = gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else open(path, encoding='utf-8')
    with opener as source:
        for line_number, line in enumerate(source, 1):
            doc_id = None
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('record must be an object')
                doc_id = record.get('doc_id')
                if not isinstance(doc_id, str) or not doc_id.strip():
                    raise ValueError('missing non-empty doc_id')
                if doc_id in seen:
                    raise ValueError(f'duplicate doc_id: {doc_id}')
                seen.add(doc_id)
                if doc_id in exclude:
                    n_excluded += 1
                    continue
                if wanted is not None and doc_id not in wanted:
                    continue
                stem = filename_stem(doc_id)
                if stem.casefold() in filenames:
                    raise ValueError(f'duplicate output filename for doc_id: {doc_id}')
                title = record.get('title')
                if not isinstance(title, str) or not title.strip():
                    raise ValueError('missing non-empty title')
                sections = record.get('sections')
                if not isinstance(sections, list):
                    raise ValueError('sections must be an array')
                abstracts = [s['text'].strip() for s in sections
                             if isinstance(s, dict) and s.get('category') == 'abstract'
                             and isinstance(s.get('text'), str) and s['text'].strip()]
                if not abstracts:
                    raise ValueError('missing non-empty abstract')
                filenames.add(stem.casefold())
                seed = {'doc_id': doc_id, 'title': title.strip(),
                        'abstract': '\n\n'.join(abstracts)}
            except (ValueError, UnicodeError) as error:
                logger.warning('Skipping %s line %d (doc_id=%r): %s',
                               path, line_number, doc_id, error)
                continue
            found.add(doc_id)
            yield seed
            count += 1
            if limit is not None and count >= limit:
                break
    if n_excluded:
        logger.info('Skipped %d excluded doc_ids in %s', n_excluded, path)
    if limit is not None and count >= limit:
        return
    if wanted is not None and wanted - found:
        logger.warning('Requested doc_ids not found or invalid in %s: %s', path, sorted(wanted - found))

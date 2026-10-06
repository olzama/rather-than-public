#!/usr/bin/env python3.12
r"""
Build one code-only "summary bundle" per cloned code-release repo: the
repo's README, a file tree, the most important source files in full, and
signature-level outlines of what didn't fit -- the input an LLM gets when
asked to describe what the code does. NO paper text is included, by
design: the summaries must describe the code as if the paper had not been
written yet.

Why a separate builder from build_code_paper_dataset.py (which is left
as-is for its own paper+code use):
  - That builder walks files in alphabetical path order and stops at
    --max-total-chars, so in the ~180 repos that hit the cap it kept
    whatever sorted first (configs, docs, JSON) and dropped entry points
    like train.py/main.py/model.py.
  - It skipped any directory named models/, data/, dataset(s)/, ... --
    which in these repos mostly holds model and data-loading *code*
    (~6k .py files across the corpus). Here, directories are only
    skipped when they never hold the authors' code (.git, node_modules,
    virtualenvs, ...); data/checkpoints are kept out by the extension
    whitelist and size/binary checks instead.

How the character budget is spent (priority order):
  tier 0  entry points: Python files with a __main__ guard or a CLI
          parser, files invoked from the README or shell scripts,
          conventionally named scripts (train/main/run/eval/...), and the
          shell scripts themselves
  tier 1  prompt templates (for LLM papers these ARE the method), and
          Python modules reachable from tier 0 through the local import
          graph, closest first
  tier 2  remaining own code
  tier 3  configs (yaml/toml/small *config*.json), sub-directory READMEs
  tier 4  tests
  tier 5  vendored code (third_party/, copies of transformers, fairseq,
          ... and files carrying a big-lab copyright header): never
          included in full, outline only
Files are taken in full, in that order, while they fit. Python (and, via
a cruder regex, other languages) that doesn't fit is reduced to an
outline -- classes, function signatures, first docstring line -- until
the outline budget runs out; anything left appears only as a path in the
file tree, which marks each file [F]ull / [O]utline / [-] omitted so the
summarizer knows what it did not see.

Usage:
    python3.12 build_code_summary_bundles.py <manifest.jsonl> <out.jsonl> \
        [--licenses licenses.json] [--render-dir DIR] [--metadata-out meta.jsonl] \
        [--max-total-chars 500000] [--max-file-chars 50000] \
        [--max-outline-chars 60000] [--max-tree-chars 15000] \
        [--focus-paths focus.json] [--only owner/repo ...]

--focus-paths (from extract_repo_focus_paths.py): for papers that link a
subdirectory of a monorepo, files under it are ranked before all others.

One record per repo (repos linked from several papers are merged; the
papers are listed in "papers"):

    {"owner_repo", "repo_url", "license", "papers": [{"corpus", "doc_id"}],
     "readme": <str or null>, "file_tree": <str>,
     "files": [{"path", "kind", "tier", "reason", "content", "truncated"}],
     "outlines": [{"path", "outline"}],
     "stats": {...coverage numbers, see collect()...}}

--render-dir writes <owner>__<repo>.txt per repo: the bundle flattened to
the plain text that is actually sent to the summarizer (see render()).

Same license caveat as build_code_paper_dataset.py: out.jsonl and the
rendered bundles embed repo content verbatim -- a private research
artifact, not something to commit or share. --metadata-out (stats only)
is safe to commit.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

if sys.version_info < (3, 9):
    sys.exit("needs Python >= 3.9 (ast.unparse, and parsing modern syntax); "
             "run with python3.12")

# Directories that never hold the authors' own code. Everything else is walked.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".env",
    "site-packages", ".ipynb_checkpoints", "wandb", ".idea", ".vscode",
    ".cache", ".mypy_cache", ".pytest_cache", ".tox", ".github", ".eggs",
}
# Directories whose contents are someone else's code bundled into the repo.
VENDOR_DIRS = {
    "third_party", "third-party", "thirdparty", "3rdparty", "external",
    "externals", "extern", "vendor", "vendored", "deps",
}
# Copies of well-known libraries, recognized by directory name.
KNOWN_LIB_DIRS = {
    "transformers", "fairseq", "apex", "megatron", "megatron-lm", "allennlp",
    "peft", "trl", "vllm", "deepspeed", "lm_eval", "lm-evaluation-harness",
    "torchtune", "nltk", "spacy", "pytorch_pretrained_bert",
    "pytorch_transformers", "detectron2", "mmcv", "timm", "open_clip",
    "sentence_transformers", "opennmt", "onmt", "tensor2tensor", "flash_attn",
    "accelerate", "datasets_lib", "llama_factory", "llamafactory", "verl",
    "openrlhf", "lavis", "fastchat",
}
VENDOR_HEADER_RE = re.compile(
    r"Copyright[^\n]{0,80}(HuggingFace|Google (AI|LLC|Inc)|The TensorFlow Authors|"
    r"Facebook|Meta Platforms|NVIDIA|Microsoft Corporation|OpenAI|Allen Institute|"
    r"The Google AI Language Team|EleutherAI|DeepMind)", re.I)

CODE_EXTS = {
    ".py", ".pyx", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".java", ".kt",
    ".scala", ".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".go", ".rs", ".rb",
    ".php", ".pl", ".pm", ".r", ".jl", ".lua", ".m", ".swift", ".hs", ".lisp",
    ".asd", ".clj", ".ml", ".mli", ".el", ".sql", ".proto", ".cs", ".fs",
    ".ex", ".exs", ".erl", ".dart", ".groovy", ".pro", ".prolog",
}
SHELL_EXTS = {".sh", ".bash", ".zsh", ".slurm", ".sbatch"}
CONFIG_EXTS = {".yaml", ".yml", ".toml", ".cfg", ".ini", ".gin", ".conf", ".jsonnet"}
PROMPT_EXTS = {".prompt", ".j2", ".jinja", ".jinja2", ".tmpl"}
SPECIAL_NAMES = {"makefile", "dockerfile", "snakefile", "justfile"}

ENTRY_NAME_RE = re.compile(
    r"^(train|main|run|eval|evaluate|infer|inference|predict|finetune|fine_tune|"
    r"pretrain|test_model|experiment|experiments|pipeline|app|demo|generate|"
    r"benchmark|score|analysis|analyze)([_\-].*)?$", re.I)
TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing|unittests?)(/|$)|(^|/)test_[^/]+\.py$|_test\.py$")
# Packaging/infrastructure/docs trees: ranked with tests, below the actual code.
INFRA_PATH_RE = re.compile(r"(^|/)(docker|docs?|ci|\.circleci|\.devcontainer|website|site)(/|$)|"
                           r"(^|/)(install|setup|build|release|stamp|sync|lint|format|run_ruff|"
                           r"publish|deploy|bump)[^/]*$", re.I)
# Order inside tier 0, strongest evidence first.
ENTRY_RANK = {"invoked from README/scripts": 0, "entry-point name": 1, "has __main__/CLI": 2,
              "run script": 3}
# Largest share of the full-text budget each non-code kind may take.
KIND_SHARE = {"shell": 0.06, "config": 0.06, "doc": 0.03}
README_CODE_IMPORT_RE = re.compile(r"^\s*(?:>>>\s*)?(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M)
CLI_RE = re.compile(r"\b(argparse|ArgumentParser|import click|@click\.|typer\.|"
                    r"@hydra\.main|fire\.Fire|absl\.app|tf\.app\.run|HfArgumentParser)\b")
MAIN_GUARD_RE = re.compile(r"""^if\s+__name__\s*==\s*['"]__main__['"]""", re.M)
# `python train.py`, `python3 -u src/run.py`, `python -m pkg.mod`, `accelerate launch x.py`,
# `torchrun ... x.py`, `deepspeed x.py`
INVOKE_PY_RE = re.compile(r"(?:^|[\s;&|(`\"'])((?:[\w./\-]+/)?[\w\-]+\.py)\b")
INVOKE_MOD_RE = re.compile(r"\bpython[\d.]*\s+(?:-\w+\s+)*-m\s+([\w.]+)")
IMPORT_RE = re.compile(r"^\s*(?:from\s+(\.*[\w.]*)\s+import\s+([\w*, ()]+)|import\s+([\w., ]+))", re.M)

SIZE_LIMIT_BYTES = 2_000_000       # larger "code" files are data or generated
CONFIG_JSON_RE = re.compile(r"(config|hparam|hyperparam|args|param|setting|ds_|deepspeed|accelerate)", re.I)


def read_text(path, max_chars=None):
    """Text content, or None for unreadable/binary files."""
    try:
        with open(path, "rb") as f:
            raw = f.read() if max_chars is None else f.read(max_chars * 4 + 4)
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    text = raw.decode("utf-8", errors="replace")
    return text if max_chars is None else text[:max_chars]


def notebook_to_text(raw):
    """Code and markdown cells of an .ipynb as '# %%' text, outputs dropped."""
    try:
        nb = json.loads(raw)
    except (ValueError, TypeError):
        return None
    cells = nb.get("cells")
    if cells is None:  # nbformat 3
        cells = [c for ws in nb.get("worksheets", []) for c in ws.get("cells", [])]
    parts = []
    for cell in cells:
        src = cell.get("source", cell.get("input", ""))
        src = "".join(src) if isinstance(src, list) else str(src)
        if not src.strip():
            continue
        if cell.get("cell_type") == "markdown":
            parts.append("# %% [markdown]\n" + "\n".join("# " + l for l in src.splitlines()))
        elif cell.get("cell_type") == "code":
            parts.append("# %%\n" + src)
    return "\n\n".join(parts)


def looks_generated(text):
    lines = text.splitlines() or [""]
    return len(text) / len(lines) > 300  # minified JS, embedded data blobs


def classify(rel, name, ext):
    """(kind, is_candidate) for a repo-relative path."""
    lower = name.lower()
    parts = rel.lower().split("/")
    if lower in SPECIAL_NAMES:
        return "shell", True
    if ext == ".ipynb":
        return "notebook", True
    if ext in SHELL_EXTS:
        return "shell", True
    if ext in CODE_EXTS:
        if lower.endswith((".min.js", ".bundle.js")):
            return None, False
        return "code", True
    if ext in PROMPT_EXTS or (ext in {".txt", ".md"} and any("prompt" in p for p in parts[:-1])):
        return "prompt", True
    if ext in CONFIG_EXTS:
        return "config", True
    if ext == ".json" and CONFIG_JSON_RE.search(name):
        return "config", True
    if lower in {"readme.md", "readme.rst", "readme.txt", "readme"} and len(parts) > 1:
        return "doc", True
    if lower in {"requirements.txt", "environment.yml", "setup.py", "pyproject.toml"}:
        return "config", True
    return None, False


def in_vendored_dir(rel):
    return any(p in VENDOR_DIRS or p in KNOWN_LIB_DIRS for p in rel.lower().split("/")[:-1])


def is_vendored(rel, kind, head):
    if in_vendored_dir(rel):
        return True
    return kind in ("code", "notebook") and bool(VENDOR_HEADER_RE.search(head[:3000]))


# ---------------------------------------------------------------- python analysis

def python_outline(text):
    """Classes/functions with signatures and first docstring line. Falls back to
    a regex scan for code ast can't parse (e.g. Python 2)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return regex_outline(text, python=True)
    out = []
    doc = ast.get_docstring(tree)
    if doc:
        out.append('"""' + doc.strip().splitlines()[0][:200] + '"""')

    def sig(fn):
        try:
            args = ast.unparse(fn.args)
        except Exception:
            args = "..."
        ret = ""
        if fn.returns is not None:
            try:
                ret = " -> " + ast.unparse(fn.returns)
            except Exception:
                pass
        prefix = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {fn.name}({args}){ret}"

    def first_doc(node):
        d = ast.get_docstring(node)
        return f'  # {d.strip().splitlines()[0][:160]}' if d and d.strip() else ""

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(sig(node) + first_doc(node))
        elif isinstance(node, ast.ClassDef):
            try:
                bases = ", ".join(ast.unparse(b) for b in node.bases)
            except Exception:
                bases = ""
            out.append(f"class {node.name}({bases}):" + first_doc(node))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append("    " + sig(sub) + first_doc(sub))
        elif isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            out.append("if __name__ == '__main__': ...")
    return "\n".join(out)


OUTLINE_RES = {
    True: re.compile(r"^\s*(class\s+\w+[^:]*:|(async\s+)?def\s+\w+\s*\([^)]*\)?)", re.M),
    False: re.compile(
        r"^\s*(?:(?:public|private|protected|static|export|async|pub|fn|func|function|def|"
        r"class|struct|interface|trait|impl|module|object|type|template|virtual|inline)\b"
        r"[^;{=\n]{0,160})", re.M),
}


def regex_outline(text, python):
    lines = [m.group(0).rstrip() for m in OUTLINE_RES[python].finditer(text)]
    return "\n".join(l[:200] for l in lines)


def module_names(rel):
    """Every dotted name a .py file might be imported as, given unknown sys.path:
    a/b/c.py -> a.b.c, b.c, c ; a/b/__init__.py -> a.b, b."""
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[i:]) for i in range(len(parts)) if parts[i:]]


def local_imports(rel, text, index):
    """Repo-relative paths of local modules imported by a .py file."""
    pkg = rel.split("/")[:-1]
    found = set()
    for m in IMPORT_RE.finditer(text):
        from_mod, names, plain = m.groups()
        targets = []
        if plain:
            targets = [t.strip().split(" as ")[0].strip() for t in plain.split(",")]
        elif from_mod is not None:
            level = len(from_mod) - len(from_mod.lstrip("."))
            base = from_mod.lstrip(".")
            if level:
                anchor = pkg[: len(pkg) - (level - 1)] if level - 1 <= len(pkg) else []
                base = ".".join(anchor + ([base] if base else []))
            names = [n.strip().split(" as ")[0].strip() for n in names.strip("() ").split(",")]
            targets = [base] + [f"{base}.{n}" if base else n for n in names if n and n != "*"]
        for t in targets:
            for path in index.get(t, ()):
                if path != rel:
                    found.add(path)
    return found


# ---------------------------------------------------------------- collection

def walk(repo_dir):
    for root, dirs, files in os.walk(repo_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs
                         if d not in SKIP_DIRS and not d.endswith(".egg-info")
                         and not os.path.islink(os.path.join(root, d)))
        for name in sorted(files):
            path = os.path.join(root, name)
            if os.path.islink(path):
                continue
            yield Path(path)


def find_readme(repo_dir, max_chars):
    for child in sorted(repo_dir.iterdir()):
        if child.is_file() and child.stem.lower() == "readme":
            return read_text(child, max_chars)
    return None


def collect(repo_dir, args, focus=()):
    readme = find_readme(repo_dir, args.max_file_chars)
    files = {}          # rel -> dict(kind, text, vendored, size)
    seen_hashes = {}
    n_dupes = n_generated = n_oversize = 0
    for path in walk(repo_dir):
        rel = path.relative_to(repo_dir).as_posix()
        if "/" not in rel and path.stem.lower() == "readme":
            continue  # captured as `readme`
        kind, ok = classify(rel, path.name, path.suffix.lower())
        if not ok:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        limit = SIZE_LIMIT_BYTES * (10 if kind == "notebook" else 1)
        if size > limit or (kind == "config" and size > 200_000):
            n_oversize += 1
            continue
        raw = read_text(path)
        if raw is None:
            continue
        text = notebook_to_text(raw) if kind == "notebook" else raw
        if not text or not text.strip():
            continue
        if kind in ("code", "notebook") and looks_generated(text):
            n_generated += 1
            continue
        digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        if digest in seen_hashes:
            n_dupes += 1
            continue
        seen_hashes[digest] = rel
        files[rel] = {"kind": kind, "text": text,
                      "vendored": is_vendored(rel, kind, text)}

    # --- entry points and invocations
    py_index = defaultdict(list)
    for rel, f in files.items():
        if rel.endswith(".py"):
            for name in module_names(rel):
                py_index[name].append(rel)
    by_basename = defaultdict(list)
    for rel in files:
        by_basename[rel.rsplit("/", 1)[-1]].append(rel)

    invoked = set()
    # Only the README and the authors' own scripts/configs count as evidence of what
    # gets run -- not CI/test/docker scripts or a vendored fork's example scripts.
    invoke_sources = [readme or ""] + [
        f["text"] for rel, f in files.items()
        if f["kind"] in ("shell", "config", "doc") and not f["vendored"]
        and not rel.rsplit("/", 1)[-1].startswith(".")  # .pre-commit-config.yaml etc.
        and not TEST_PATH_RE.search(rel) and not INFRA_PATH_RE.search(rel)]
    for src in invoke_sources:
        for m in INVOKE_PY_RE.finditer(src):
            ref = m.group(1).lstrip("./")
            if ref in files:
                invoked.add(ref)
            else:  # scripts often cd into a subdir first; match by path suffix
                for rel in by_basename.get(ref.rsplit("/", 1)[-1], ()):
                    if rel.endswith(ref):
                        invoked.add(rel)
        for m in INVOKE_MOD_RE.finditer(src):
            for rel in py_index.get(m.group(1), ()):
                invoked.add(rel)

    tier, reason = {}, {}
    for rel, f in files.items():
        stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        is_test = bool(TEST_PATH_RE.search(rel) or INFRA_PATH_RE.search(rel))
        if rel in invoked and not is_test:  # even inside a vendored fork: the authors run it
            tier[rel], reason[rel] = 0, "invoked from README/scripts"
        elif f["vendored"]:
            tier[rel], reason[rel] = 5, "vendored"
        elif f["kind"] == "shell":
            tier[rel], reason[rel] = 0, "run script"
        elif f["kind"] in ("code", "notebook") and not is_test and (
                MAIN_GUARD_RE.search(f["text"]) or CLI_RE.search(f["text"])):
            tier[rel], reason[rel] = 0, "has __main__/CLI"
        elif f["kind"] in ("code", "notebook") and not is_test and ENTRY_NAME_RE.match(stem):
            tier[rel], reason[rel] = 0, "entry-point name"
        elif f["kind"] == "prompt":
            tier[rel], reason[rel] = 1, "prompt template"
        elif f["kind"] in ("config", "doc"):
            tier[rel], reason[rel] = 3, f["kind"]
        elif is_test:
            tier[rel], reason[rel] = 4, "test"
        else:
            tier[rel], reason[rel] = 2, "other code"

    # --- import graph BFS from entry points: reachable modules move to tier 1.
    # Local modules imported in README code examples (library-style repos, where
    # usage is `from pkg import X` rather than a script) seed the search too.
    dist = {rel: 0 for rel in files if tier[rel] == 0 and rel.endswith(".py")}
    for m in README_CODE_IMPORT_RE.finditer(readme or ""):
        for rel in py_index.get(m.group(1) or m.group(2), ()):
            if rel not in dist and not files[rel]["vendored"]:
                dist[rel] = 1
                if tier[rel] in (2, 4):
                    tier[rel], reason[rel] = 1, "imported in README example"
    queue = deque(dist)
    while queue:
        cur = queue.popleft()
        for dep in local_imports(cur, files[cur]["text"], py_index):
            if dep not in dist and not files[dep]["vendored"]:
                dist[dep] = dist[cur] + 1
                queue.append(dep)
                if tier[dep] == 2:
                    tier[dep], reason[dep] = 1, f"imported by entry point (depth {dist[dep]})"

    def in_focus(rel):
        return any(rel == f or rel.startswith(f + "/") for f in focus)

    # A linked top-level directory is a project folder in a monorepo: the rest of
    # the repo is other projects, so it is left out (file tree only). A linked file
    # or nested path just marks where to start; the rest is still the method.
    restrict = bool(focus) and all("/" not in f and (repo_dir / f).is_dir() for f in focus)

    def sort_key(rel):
        return (not in_focus(rel), tier[rel], ENTRY_RANK.get(reason[rel], 0), dist.get(rel, 99),
                rel.count("/"), len(files[rel]["text"]), rel)

    order = sorted(files, key=sort_key)

    # --- fill budget: full text first, outlines for what doesn't fit
    tree_reserve = args.max_tree_chars
    total_source = sum(len(f["text"]) for f in files.values())
    outline_reserve = min(args.max_outline_chars, max(0, total_source - args.max_total_chars))
    if focus and all("/" not in f and (repo_dir / f).is_dir() for f in focus):
        in_scope = sum(len(f["text"]) for r, f in files.items()
                       if any(r.startswith(x + "/") for x in focus))
        outline_reserve = min(args.max_outline_chars, max(0, in_scope - args.max_total_chars))
    full_budget = args.max_total_chars - tree_reserve - outline_reserve - len(readme or "")
    status = {}
    out_files, pending_outline = [], []
    used = 0
    kind_used = defaultdict(int)
    config_cap = args.max_file_chars // 5
    for rel in order:
        f = files[rel]
        if restrict and not in_focus(rel):
            status[rel] = "-"
            continue
        if tier[rel] == 5:
            pending_outline.append(rel)
            continue
        cap = config_cap if f["kind"] in ("config", "doc") else args.max_file_chars
        text = f["text"]
        truncated = len(text) > cap
        if truncated:
            text = text[:cap] + f"\n... [truncated: {len(f['text']) - cap} more chars]"
        share = KIND_SHARE.get(f["kind"])
        if share is not None and kind_used[f["kind"]] + len(text) > share * full_budget:
            status[rel] = "-"
            continue
        if used + len(text) <= full_budget:
            kind_used[f["kind"]] += len(text)
            out_files.append({"path": rel, "kind": f["kind"], "tier": tier[rel],
                              "reason": reason[rel], "content": text, "truncated": truncated})
            used += len(text)
            status[rel] = "F"
        else:
            pending_outline.append(rel)

    outlines, outline_used = [], 0
    outline_budget = args.max_total_chars - tree_reserve - used - len(readme or "")
    for rel in pending_outline:
        f = files[rel]
        if restrict and not in_focus(rel):
            status[rel] = "-"
            continue
        if f["kind"] not in ("code", "notebook"):
            status[rel] = "-"
            continue
        if rel.endswith(".py") or f["kind"] == "notebook":
            outline = python_outline(f["text"])
        else:
            outline = regex_outline(f["text"], python=False)
        if outline and outline_used + len(outline) <= outline_budget:
            outlines.append({"path": rel, "outline": outline})
            outline_used += len(outline)
            status[rel] = "O"
        else:
            status[rel] = "-"

    file_tree = render_tree(files, status, tier, args.max_tree_chars)
    full_chars = sum(len(files[r]["text"]) for r, s in status.items() if s == "F")
    stats = {
        "focus_paths": list(focus),
        "n_candidate_files": len(files),
        "n_full": sum(s == "F" for s in status.values()),
        "n_outline": sum(s == "O" for s in status.values()),
        "n_listed_only": sum(s == "-" for s in status.values()),
        "n_vendored": sum(f["vendored"] for f in files.values()),
        "n_duplicates_skipped": n_dupes,
        "n_generated_skipped": n_generated,
        "n_oversize_skipped": n_oversize,
        "entry_points": sorted(r for r in files if tier[r] == 0 and files[r]["kind"] != "shell"),
        "source_chars_total": total_source,
        "full_chars": used,
        "outline_chars": outline_used,
        "own_code_full_fraction": round(
            full_chars / max(1, sum(len(f["text"]) for f in files.values() if not f["vendored"])), 3),
        "focus_restricted": restrict,
        "complete": all(s == "F" for r, s in status.items()
                        if tier[r] != 5 and (not restrict or in_focus(r))),
    }
    return readme, file_tree, out_files, outlines, stats


def render_tree(files, status, tier, max_chars):
    """Directory listing with per-file status marks; directories whose files were
    all omitted collapse to one line so huge vendored trees don't eat the budget."""
    by_dir = defaultdict(list)
    for rel in sorted(files):
        d, _, name = rel.rpartition("/")
        by_dir[d].append((name, rel))
    lines = []
    for d in sorted(by_dir):
        entries = by_dir[d]
        depth = d.count("/") + 1 if d else 0
        indent = "  " * depth
        if d:
            lines.append("  " * (depth - 1) + d.rsplit("/", 1)[-1] + "/")
        if len(entries) > 12 and all(status.get(r) == "-" for _, r in entries):
            lines.append(f"{indent}[-] ({len(entries)} files not shown"
                         f"{', vendored' if all(tier[r] == 5 for _, r in entries) else ''})")
            continue
        for name, rel in entries:
            vend = " (vendored)" if tier[rel] == 5 else ""
            lines.append(f"{indent}[{status.get(rel, '-')}] {name}{vend}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [file tree truncated]"
    return text


def render(record):
    """Flatten a bundle to the plain text handed to the summarizer. XML-style tags
    delimit sections so they can't be confused with markdown/comments in the code."""
    s = record["stats"]
    out = [f'<repository name="{record["owner_repo"]}">',
           f"<coverage>{s['n_full']} files in full, {s['n_outline']} as outlines, "
           f"{s['n_listed_only']} listed only (of {s['n_candidate_files']} code/config files; "
           f"{s['n_vendored']} vendored). File tree marks: [F] full, [O] outline, [-] not shown."
           + (f" Focus: the project's code lives under {', '.join(s['focus_paths'])}; "
              "the rest of the repository is unrelated or shared infrastructure."
              if s.get("focus_paths") else "") + "</coverage>",
           "<readme>", record["readme"] or "(no README)", "</readme>",
           "<file_tree>", record["file_tree"] or "(no code files)", "</file_tree>"]
    for f in record["files"]:
        out += [f'<file path="{f["path"]}">', f["content"], "</file>"]
    for o in record["outlines"]:
        out += [f'<outline path="{o["path"]}">', o["outline"], "</outline>"]
    out.append("</repository>")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("out_path")
    p.add_argument("--licenses", default=None)
    p.add_argument("--render-dir", default=None)
    p.add_argument("--metadata-out", default=None)
    p.add_argument("--max-total-chars", type=int, default=500_000)
    p.add_argument("--max-file-chars", type=int, default=50_000)
    p.add_argument("--max-outline-chars", type=int, default=60_000)
    p.add_argument("--max-tree-chars", type=int, default=15_000)
    p.add_argument("--focus-paths", default=None,
                   help="extract_repo_focus_paths.py output ('owner/repo' -> [subpaths]); "
                        "files under these paths are ranked before everything else")
    p.add_argument("--only", nargs="*", default=None, help="owner/repo names to build")
    args = p.parse_args()

    licenses = json.load(open(args.licenses)) if args.licenses else {}
    focus_paths = json.load(open(args.focus_paths)) if args.focus_paths else {}
    repos = {}  # owner_repo -> {"local_dir", "papers"}
    for line in open(args.manifest):
        row = json.loads(line)
        if row["clone_status"] != "ok":
            continue
        entry = repos.setdefault(row["owner_repo"], {"local_dir": row["local_dir"], "papers": []})
        entry["papers"].append({"corpus": row["corpus"], "doc_id": row["doc_id"]})
    if args.only:
        repos = {k: v for k, v in repos.items() if k in set(args.only)}
    if args.render_dir:
        os.makedirs(args.render_dir, exist_ok=True)

    meta_out = open(args.metadata_out, "w") if args.metadata_out else None
    n_written = n_missing = 0
    with open(args.out_path, "w") as out:
        for i, (owner_repo, entry) in enumerate(sorted(repos.items()), 1):
            repo_dir = Path(entry["local_dir"])
            if not repo_dir.is_dir():
                n_missing += 1
                continue
            readme, tree, files, outlines, stats = collect(repo_dir, args, focus_paths.get(owner_repo, ()))
            record = {"owner_repo": owner_repo, "repo_url": f"https://github.com/{owner_repo}",
                      "license": licenses.get(owner_repo, "unknown"), "papers": entry["papers"],
                      "readme": readme, "file_tree": tree, "files": files,
                      "outlines": outlines, "stats": stats}
            out.write(json.dumps(record) + "\n")
            if meta_out:
                meta_out.write(json.dumps({k: record[k] for k in
                               ("owner_repo", "repo_url", "license", "papers", "stats")}) + "\n")
            if args.render_dir:
                with open(os.path.join(args.render_dir, owner_repo.replace("/", "__") + ".txt"), "w") as f:
                    f.write(render(record))
            n_written += 1
            if i % 50 == 0:
                print(f"  {i}/{len(repos)}", file=sys.stderr)
    if meta_out:
        meta_out.close()
    print(f"DONE {n_written} repos -> {args.out_path} ({n_missing} missing dirs)", file=sys.stderr)


if __name__ == "__main__":
    main()

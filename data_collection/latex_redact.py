#!/usr/bin/env python3
r"""
Replace LaTeX math and \ref/\cite-family commands with a single,
unambiguous placeholder instead of letting detex delete them (silent,
leaves a grammatically broken gap) or substitute its own "noun"/"noun
verbs noun" for math (confusing specifically for this corpus: these are
NLP papers, where "noun" is itself a common content word -- "noun
phrase", "proper noun" -- so the placeholder was indistinguishable from
real text; found 2026-09-14).

Two passes:

1. \ref/\cite-family commands are redacted with a direct regex on the
   raw .tex source, before detex ever runs. Safe to do by hand: the
   argument is delimited by a single non-nested {...}, so the regex
   can't run away -- it just under-matches (falls back to detex's normal
   silent deletion) if a label/key ever contains a nested brace, rare.

2. Math ($...$, \[...\], environments) is NOT hand-parsed at the source
   level -- a first version tried a $...$ regex and it silently ate
   ~17KB of one real paper (2601.00263) after a single unescaped "$"
   elsewhere in the document made the regex treat two unrelated math
   spans as one. A second version ran detex twice (plain and -r) and
   diffed the outputs to relabel -r's "noun" placeholders, which was
   correct but too slow at corpus scale (difflib.SequenceMatcher with
   autojunk off took ~17s on one ~24K-token document; autojunk on was
   fast but imprecise, swallowing real words like "and" that sat between
   two redactions into a single merged placeholder).

   Math is instead redacted by post-processing PLAIN detex's own output
   (no -r): detex already reliably deletes math with no artifact-
   specific corruption (that's what makes the plain run safe to trust),
   leaving a run of 2+ literal spaces exactly where each span was
   removed (single spaces from both sides of the deleted span, with
   nothing between them) -- e.g. "on $X$ ensures" -> "on  ensures".
   Collapsing every such between-word gap to " [redacted] " is the same
   signal this project already used to detect the corruption in the
   first place, reused here as the fix; it under-fires only for a math
   span with no surrounding whitespace at all (e.g. "value$X$,"), same
   as any of the approaches above.
"""
import re
import shutil
import subprocess
from pathlib import Path

PLACEHOLDER = "[redacted]"

_REF_RE = re.compile(r"\\(?:[Cc]ref|eqref|autoref|pageref|nameref|ref)\*?\{[^{}]*\}")
_CITE_RE = re.compile(r"\\[Cc]ite\w*\*?(?:\[[^\]]*\]){0,2}\{[^{}]*\}")
_GAP_RE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")


def redact_refs_and_cites(text):
    text = _REF_RE.sub(PLACEHOLDER, text)
    text = _CITE_RE.sub(PLACEHOLDER, text)
    return text


def redact_source_tree(src_dir, work_dir):
    """Copy src_dir to work_dir and redact \\ref/\\cite in every .tex file
    in place. Returns work_dir. detex can still follow \\input/\\include
    within work_dir exactly as it would in src_dir, just reading
    ref/cite-redacted text -- PROVIDED it's invoked with cwd=work_dir (or
    the main .tex file's own directory within it); see run_detex, which
    enforces this. math is handled separately, on detex's plain output
    (see redact_math_gaps)."""
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(src_dir, work_dir)
    for f in work_dir.rglob("*.tex"):
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        f.write_text(redact_refs_and_cites(text))
    return work_dir


def redact_math_gaps(detex_plain_output):
    """Collapse every run of 2+ spaces/tabs between two words in PLAIN
    detex's own output into " [redacted] " -- see module docstring for
    why this, rather than parsing math directly, is the reliable signal
    for "detex deleted a math span here"."""
    return _GAP_RE.sub(f" {PLACEHOLDER} ", detex_plain_output)


_INPUT_FAIL_RE = re.compile(r"can't open \\input file")


def run_detex(main_tex):
    """Run detex on main_tex, correctly resolving any \\input/\\include it
    contains -- which needs subprocess's cwd set to main_tex's own
    directory. detex resolves \\input paths relative to its OWN current
    working directory, not relative to the file it was told to process;
    call it from anywhere else (e.g. the CWD of whatever script invoked
    this) and every \\input silently fails with a stderr warning while
    still exiting 0, so the pipeline has no way to notice from the
    return code alone -- found 2026-09-15 after ~30% of one corpus came
    back as near-empty extractions (just the title/author block, the
    entire body silently missing wherever it lived in an \\input'd file).

    Returns (stdout, ok, had_input_failure). ok is False on a nonzero
    exit (a real detex failure, e.g. unparseable LaTeX); had_input_failure
    is True if detex ran fine (exit 0) but still couldn't open at least
    one \\input/\\include target (e.g. a genuinely missing file, as
    opposed to the cwd bug this function exists to prevent) -- a case
    worth logging even though it isn't fatal, since it means partial
    content loss for a different reason than the one this function fixes.
    """
    main_tex = Path(main_tex)
    result = subprocess.run(
        ["detex", main_tex.name], cwd=main_tex.parent, capture_output=True, text=True,
    )
    had_input_failure = bool(_INPUT_FAIL_RE.search(result.stderr))
    return result.stdout, result.returncode == 0, had_input_failure

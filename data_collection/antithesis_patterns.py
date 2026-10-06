r"""
Shared pattern list for the antithesis-family construction counters
(../counting/count_antithesis_patterns.py, extract_antithesis_instances.py,
here). Kept in one place so the two scripts can't drift apart -- they did
once already (a "not only" exclusion was correct in one place and
silently dropped matches in the other before the two were unified).

Tier taxonomy, agreed with the user 2026-09-14: Tier 1 is direct
syntactic siblings of "rather than" (explicit substitution connectives);
Tier 2 is canonical negate-then-affirm rhetorical antithesis; Tier 3 is
emphatic pivot constructions, the closest match to the paper's actual
complaint (LLM writing that states what it does NOT do before saying
what it does).
"""
import re

# (id, tier, label, regex, note)
PATTERNS = [
    # --- Tier 1: direct syntactic siblings of "rather than" ---
    ("rather_than", 1, "rather than",
     r"\brather than\b", None),
    ("instead_of", 1, "instead of",
     r"\binstead of\b", None),
    ("as_opposed_to", 1, "as opposed to",
     r"\bas opposed to\b", None),
    ("in_contrast", 1, "in contrast (to)",
     r"\bin contrast\b", "covers both \"in contrast to X\" and sentence-initial \"In contrast, ...\"; "
     "EXCLUDED from the paper: mostly a discourse connective whose contrasted alternative is in the previous "
     "sentence and is often not rejected (human spot check, data/pattern_precision)"),

    # --- Tier 2: canonical negate-then-affirm antithesis ---
    # Excludes "not only/just/merely ... but" so it doesn't double-count Tier 3's
    # emphatic variant. Bounded to a same-sentence window (no ./!/? in between).
    ("not_but", 2, "not X but Y",
     r"\bnot\b(?!\s+(?:only|just|merely)\b)[^.!?]{0,60}?\bbut\b", "same-sentence window, excludes not-only/just/merely"),
    # Very noisy: a comma followed by "not" is common for reasons that have
    # nothing to do with antithesis (e.g. "..., not knowing what to do, she...").
    ("x_not_y", 2, "X, not Y (postposed)",
     r",\s*not\b(?!\s+(?:only|just|merely)\b)", "high recall, low precision -- see caveats"),

    # --- Tier 3: emphatic pivot constructions ---
    ("cleft_pivot", 3, "it's not X, it's Y",
     r"\bit'?s\s+not\b[^.!?]{0,60}?\bit'?s\b|\bit\s+is\s+not\b[^.!?]{0,60}?\bit\s+is\b", None),
    ("not_only_just_merely_but", 3, "not only/just/merely X but Y",
     r"\bnot\s+(?:only|just|merely)\b[^.!?]{0,60}?\bbut\b", None),
    ("not_to_say", 3, "this is not to say",
     r"\bthis is not to say\b", None),
]


def compile_patterns():
    return {pid: re.compile(pattern, re.I) for pid, _tier, _label, pattern, _note in PATTERNS}


def pattern_meta():
    return {pid: {"tier": tier, "label": label, "note": note} for pid, tier, label, _pattern, note in PATTERNS}

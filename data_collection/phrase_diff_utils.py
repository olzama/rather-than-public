#!/usr/bin/env python3
"""
Shared logic for classifying a text change (git diff hunk, or a v1-vs-latest
diff opcode) with respect to a target phrase like "rather than", used by
both version_history/mine_edit_history.py and arxiv_versions/diff_versions.py.

The naive approach -- check each diff line individually for the phrase --
has a real bug: prose in LaTeX source (or its detex'd text) wraps at
whatever column the author's editor used, so an edit earlier in a paragraph
commonly shifts where "rather than" falls across a line break in the *new*
text even though the phrase itself is untouched (e.g. "...multiplication,
rather\nthan a custom kernel..."). Checking line-by-line makes that a false
"removed" hit. The fix is to join each side's lines into one string before
matching.

This module also implements the finer distinction a phrase disappearing
from a hunk doesn't answer on its own: was "X" in "X rather than Y" kept
(a targeted edit that specifically cut the antithesis clause) or did the
whole surrounding sentence/paragraph get rewritten (the phrase's disappearance
is incidental)? We call the former "surgical": the several words immediately
before the phrase in the old text reappear verbatim in the new text.
"""
import re


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def join_lines(lines):
    return normalize(" ".join(lines))


def classify_change(old_text, new_text, phrase_re, context_words=6, search_words=15):
    """
    old_text / new_text: normalized (already whitespace-joined) strings for
    the two sides of a change.

    Returns one of:
      "added"          -- phrase in new_text only
      "removed"        -- phrase in old_text only, and the words immediately
                           preceding it do NOT reappear in new_text (i.e. the
                           surrounding sentence was rewritten too -- the
                           phrase's disappearance is incidental, not a
                           targeted cut)
      "removed_surgical" -- phrase in old_text only, and the clause preceding
                           it DOES reappear in new_text: a clean "X rather
                           than Y" -> "X ..." edit
      "reworded"        -- phrase present on both sides (kept, though the
                           text around it may have changed)
      "none"            -- phrase in neither (caller shouldn't normally see this)
    """
    old_has = bool(phrase_re.search(old_text))
    new_has = bool(phrase_re.search(new_text))

    if old_has and new_has:
        return "reworded"
    if new_has and not old_has:
        return "added"
    if not old_has:
        return "none"

    # old_has and not new_has: phrase was removed. Check whether the
    # clause leading up to it survived, to distinguish a targeted cut
    # from an incidental disappearance amid a larger rewrite.
    #
    # This isn't just "do the N words right before the phrase survive":
    # sometimes a few words immediately adjacent to "rather than" go with
    # it (e.g. a parenthetical that only existed to set up the
    # comparison -- "stored in inference-transposed layout ($H \times r$
    # rather than $r \times H$ for $A$)" keeps "stored in
    # inference-transposed layout" but drops the parenthetical too, so
    # the surviving clause isn't flush against the cut point). We
    # therefore slide both the window length (3..context_words) and its
    # start position across the last `search_words` words before the
    # phrase, and accept a match at any position. Trailing punctuation is
    # stripped since it commonly changes with whatever followed (a comma
    # before "rather than X," vs. a period once X is gone).
    m = phrase_re.search(old_text)
    before = old_text[:m.start()]
    words = before.split()
    region = words[-search_words:]
    new_lower = new_text.lower()
    # A minimum of 3 words produced false positives on generic short
    # phrases ("on the holdout") that coincidentally survive by accident,
    # not because the real antithesis clause did. 5 is much less likely
    # to occur by chance in prose.
    min_words = min(5, len(region))
    for n in range(min(context_words, len(region)), min_words - 1, -1):
        for start in range(0, len(region) - n + 1):
            snippet = " ".join(region[start:start + n]).strip(",.;:()")
            if snippet and snippet.lower() in new_lower:
                return "removed_surgical"
    return "removed"

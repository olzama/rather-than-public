# Annotation guideline: legitimate vs. annoying "rather than"

For each sentence, the highlighted **"rather than"** sits inside a real sentence
pulled from a real paper (2019 ACL Anthology or 2026 arXiv NLP papers). Judge
only this specific use of the construction.

## Labels

- **legitimate** — "rather than X" does real work: it names a genuine
  alternative the reader would otherwise assume, clarifies a design/methodological
  choice, or draws a contrast that matters for understanding the claim. Test: if
  you deleted "rather than X", would the sentence lose information the reader
  needs? If yes → legitimate.

- **annoying** — this use of "rather than" annoys you. Go with your reaction;
  there is no need to work out why.

- **unsure** — genuinely ambiguous, or the sentence assumes context you don't
  have (e.g. it clearly refers to something earlier) and you can't tell either way.

- **garbled** — the extracted text itself is broken (PDF column-merge,
  table/figure text mixed into the sentence, mid-word line breaks) badly enough
  that you can't actually read the sentence as written English. Not a judgment
  about the antithesis — a judgment about extraction quality.

## What NOT to judge

Not the paper's quality, not whether you agree with the claim, not grammar
elsewhere in the sentence. Only: does *this* "rather than" earn its place.

## Calibration examples

**legitimate** (real, from the sample):
> "When a paper reports multiple GPU configurations, we use the largest
> configuration rather than summing across all configurations."

A real methodological choice between two concrete alternatives; the reader
needs to know which one was taken.

> "Several widely used video benchmarks follow the same distribution model,
> providing pre-extracted features, annotations, and evaluation tooling rather
> than raw video files."

Names a specific, non-obvious alternative (raw video files) that a reader
might otherwise assume was provided.

**annoying** (constructed illustration — the real sample may have few or many
like this; that's part of what we're measuring):
> "Our approach focuses on producing accurate, reliable results rather than
> inaccurate, unreliable ones."

## Format

One label per item, independently — don't look at other items' labels or try
to guess the source paper/era.

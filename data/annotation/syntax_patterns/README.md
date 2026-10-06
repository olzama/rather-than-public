Output of `counting/syntax_patterns.py`: every tested syntactic pattern with its
count among annoying and legitimate items, Fisher p and Benjamini-Hochberg q.
`pool_<annotator>.tsv`: arXiv 2026 pool items (patterns in >= 5 items);
`pool_<annotator>_batch<N>.tsv`: one annotation batch (--batch N --min 1, all
patterns). `Either`: annoying if either annotator labeled it so, legitimate if
both did. GPT comparison set: `../../annotation_b3/syntax_patterns/` (--stratify).

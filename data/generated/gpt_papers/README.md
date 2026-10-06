# LLM-written papers (dataset 3)

Papers written by three OpenAI models (gpt-4o, gpt-5.6-sol, gpt-6-sol) from the
title and abstract of real papers. Each model writes a paper, a reviewer model
(gpt-4o) reviews it, and the same model revises it in response to the review.

| Archive | Source papers | Per model |
|---|---|---|
| `batch2.zip` | 50 ACL 2019, 50 arXiv 2026 | 100 papers |
| `batch3.zip` | 150 ACL 2019, 150 arXiv 2026 | 300 papers |
| Total | 200 ACL 2019, 200 arXiv 2026 | 400 papers |

Every model writes from the same source papers. The source papers of every batch
are listed in `paper_pipeline/used_papers.tsv`; no source paper occurs in two
batches. Batch 1 of the registry was a pilot run and is not used.

Unzip in this folder:

    unzip batch2.zip && unzip batch3.zip

Each run folder holds `<model>/{generated,evaluated,revised}_papers/<paper_id>.md`
(`evaluated_papers` are the reviews) and `<model>/sample_ids.tsv`. The scripts that
read these papers take any number of run folders and combine them
(`counting/generation_view.py`), e.g. from `counting/`:

    python3 generation_rates.py ../data/generated/gpt_papers/batch2 ../data/generated/gpt_papers/batch3 \
        --acl-dir ../data/acl2019/text_body --arxiv-dir ../data/arxiv2026/text_body \
        --excluded ../data/excluded_documents.tsv

Generation code and settings: `paper_pipeline/`.

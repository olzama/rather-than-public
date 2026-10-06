#!/usr/bin/env bash
# Re-run every analysis behind the paper's annotation numbers and write one
# report, to update the paper after new labels come in (e.g. the end of
# batch 2). Run from counting/ after pulling labels
# (data_collection/pull_sheet_labels.py). Analyses that need the embeddings
# cache or an API key (cohesion, LLM judge) are listed at the end, not run.
#
# Usage: bash refresh_annotation_numbers.sh [BATCH] > report.txt
#   BATCH: 1, 2, or empty for all batches (default: all)
set -u
A=../data/annotation
B=${1:-}
BATCH_OPT=${B:+--batch $B}

section() { printf '\n==================== %s ====================\n' "$1"; }

section "Rates per stratum, drift, order (per annotator; all batches)"
for n in A1 A2; do python3 annotation_report.py $A --annotator $n; done

section "Rates, agreement, both/either band ${B:+(batch $B)}"
python3 multi_annotator.py $A --pair A1 A2 $BATCH_OPT

section "Encounter model ${B:+(batch $B)}"
python3 encounter_model.py $A ../data/antithesis_instances --papers acl2019=4863 arxiv2026=1841 \
    --exclude ../data/excluded_documents.tsv $BATCH_OPT

section "Research domain ${B:+(batch $B)}"
python3 domain_test.py $A $A/paper_domains.jsonl --pair A1 A2 $BATCH_OPT

section "AI tells ${B:+(batch $B)}"
python3 ai_tells_test.py $A --pair A1 A2 $BATCH_OPT

section "Clustering by paper ${B:+(batch $B)}"
python3 cluster_test.py $A $BATCH_OPT

section "Straw men vs annoyance, inter-coder kappa (straw-man table)"
python3 strawman_association.py $A

section "Evaluative stance (stance table; scores from stance_llm.py)"
python3 stance_analysis.py $A $A/stance/stance__gpt-6-sol.jsonl $BATCH_OPT

section "Carry-over after blatant straw men"
python3 carryover_test.py $A --window 5

section "Syntax: all parse patterns vs. annoyance (per annotator; arXiv 2026 items)"
for n in A1 A2; do echo "-- $n"; python3 syntax_patterns.py $A --annotator $n --top 15 2>/dev/null; done

section "Not run here (need the embeddings cache / API key)"
echo "group_cohesion.py, semantic_similarity.py, egregious_similarity.py (--cache ...), llm_judge_report.py"
echo "Either-label versions of the correlates: build a derived human_labels.jsonl with an 'either' annotator"

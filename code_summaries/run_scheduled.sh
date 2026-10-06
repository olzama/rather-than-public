#!/bin/bash
# Runs summarize_code_bundles.py in batches of $BATCH repos, sleeping between
# batches so each batch stays within the calibrated ~30% of the 5h usage
# window (measured: ~0.6%/repo -> 50 repos/batch), until the whole selection
# is summarized. Detached from the shell that launched it (setsid+nohup+disown)
# so it keeps running after this terminal/session ends.
#
# Safe to interrupt/kill at any point: summarize_code_bundles.py skips repos
# that already have a .md file, so restarting this script (or rerunning the
# python command directly) just resumes.
set -u
cd "$(dirname "$0")"
SEL=../../data/code_release_repos/code_summary_selection.txt
BUNDLES=../../data/code_release_repos/code_summary_bundles
OUT=../../data/code_release_repos/code_summaries
LOG="$OUT/scheduler.log"
BATCH=50
SLEEP_SECONDS=$((5*3600 + 10*60))  # 5h10m: 5h window + margin

total=$(wc -l < "$SEL")
mkdir -p "$OUT"
echo "$(date -Is) scheduler started, pid $$, batch=$BATCH sleep=${SLEEP_SECONDS}s" >> "$LOG"

while true; do
    done_n=$(ls "$OUT"/*.md 2>/dev/null | wc -l)
    remaining=$((total - done_n))
    if [ "$remaining" -le 0 ]; then
        echo "$(date -Is) all done ($done_n/$total)" >> "$LOG"
        break
    fi
    echo "$(date -Is) batch start: done=$done_n remaining=$remaining" >> "$LOG"
    python3 summarize_code_bundles.py "$SEL" "$BUNDLES" "$OUT" --workers 4 --limit "$BATCH" >> "$LOG" 2>&1
    done_n=$(ls "$OUT"/*.md 2>/dev/null | wc -l)
    remaining=$((total - done_n))
    if [ "$remaining" -le 0 ]; then
        echo "$(date -Is) all done ($done_n/$total)" >> "$LOG"
        break
    fi
    echo "$(date -Is) batch end: done=$done_n remaining=$remaining; sleeping ${SLEEP_SECONDS}s" >> "$LOG"
    sleep "$SLEEP_SECONDS"
done
echo "$(date -Is) scheduler exiting" >> "$LOG"

#!/usr/bin/env bash
# Download PDFs for a list of ACL Anthology IDs and extract plain text.
#
# Usage:
#   ./download_acl2019.sh <ids_file> <out_dir>
#
# <ids_file> is one Anthology ID per line (see extract_acl2019_ids.py).
# <out_dir> will get pdfs/ and text/ subdirectories.
set -uo pipefail

IDS_FILE="${1:?usage: $0 <ids_file> <out_dir>}"
OUT_DIR="${2:?usage: $0 <ids_file> <out_dir>}"
OUT_PDF="$OUT_DIR/pdfs"
OUT_TXT="$OUT_DIR/text"
LOG="$OUT_DIR/download.log"

mkdir -p "$OUT_PDF" "$OUT_TXT"
: > "$LOG"

total=$(wc -l < "$IDS_FILE")
count=0
ok=0
fail=0

while IFS= read -r id; do
  count=$((count+1))
  pdf="$OUT_PDF/$id.pdf"
  txt="$OUT_TXT/$id.txt"
  if [ -s "$txt" ]; then
    ok=$((ok+1))
    continue
  fi
  url="https://aclanthology.org/${id}.pdf"
  if curl -sf --max-time 30 -o "$pdf" "$url"; then
    # NOT -layout: on two-column PDFs (nearly all ACL papers), -layout preserves
    # raw left/right visual position, so a left-column line and a right-column
    # line at the same height get concatenated into one garbled line. Default
    # (reading-order) mode handles multi-column layout correctly. Verified
    # against 300 sampled files: -layout produced this artifact in 297/300.
    if pdftotext "$pdf" "$txt" 2>>"$LOG"; then
      ok=$((ok+1))
    else
      fail=$((fail+1))
      echo "PDFTOTEXT_FAIL $id" >> "$LOG"
    fi
  else
    fail=$((fail+1))
    echo "DOWNLOAD_FAIL $id" >> "$LOG"
    rm -f "$pdf"
  fi
  if [ $((count % 100)) -eq 0 ]; then
    echo "$(date +%H:%M:%S) progress: $count/$total ok=$ok fail=$fail" >> "$LOG"
  fi
  sleep 0.25
done < "$IDS_FILE"

echo "$(date +%H:%M:%S) DONE total=$total ok=$ok fail=$fail" >> "$LOG"

#!/usr/bin/env bash

# Time the paper pipeline while preserving its output and exit status.
set -uo pipefail

export PYTHONPATH="${PYTHONPATH_OVERRIDE:-paper_pipeline/src}"

command=(
  python3 -m paper_pipeline.main
  --num-papers 1
  --config paper_pipeline/config/config.yaml
)

# Optional: provide a different command after the script name.
if (( $# > 0 )); then
  command=("$@")
fi

resource_file="$(mktemp)"
trap 'rm -f "$resource_file"' EXIT

start_iso="$(date --iso-8601=ns)"
start_ns="$(date +%s%N)"

printf 'Command: '
printf '%q ' "${command[@]}"
printf '\nStarted: %s\n\n' "$start_iso"

if [[ -x /usr/bin/time ]]; then
  /usr/bin/time \
    -f $'User CPU time: %U s\nSystem CPU time: %S s\nCPU usage: %P\nPeak memory: %M KiB' \
    -o "$resource_file" \
    -- "${command[@]}"
  exit_code=$?
else
  "${command[@]}"
  exit_code=$?
fi

end_ns="$(date +%s%N)"
end_iso="$(date --iso-8601=ns)"
elapsed_ns=$((end_ns - start_ns))
elapsed_seconds=$((elapsed_ns / 1000000000))
remaining_ns=$((elapsed_ns % 1000000000))
hours=$((elapsed_seconds / 3600))
minutes=$(((elapsed_seconds % 3600) / 60))
seconds=$((elapsed_seconds % 60))

printf '\n--- Timing summary ---\n'
printf 'Started:      %s\n' "$start_iso"
printf 'Finished:     %s\n' "$end_iso"
printf 'Elapsed:      %02d:%02d:%02d.%09d\n' \
  "$hours" "$minutes" "$seconds" "$remaining_ns"
printf 'Elapsed (s):  %d.%09d\n' "$elapsed_seconds" "$remaining_ns"
printf 'Elapsed (ns): %d\n' "$elapsed_ns"
printf 'Exit code:    %d\n' "$exit_code"

if [[ -s "$resource_file" ]]; then
  printf '\n--- Resource usage ---\n'
  sed 's/^/  /' "$resource_file"
fi

exit "$exit_code"

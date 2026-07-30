#!/bin/bash
# Launch an interactive CPU-only session via salloc + srun --pty bash.
#
# Usage:
#   bash scripts/slurm/interactive_cpu.sh                # no time limit (partition default)
#   bash scripts/slurm/interactive_cpu.sh 0-02:00:00      # cap at 2 hours
#
# Adjust cpus/mem below to taste before running.

set -euo pipefail

SALLOC_ARGS=(
  --job-name=interactive_cpu
  --account=research
  --cpus-per-task=4
  --mem=16G
)

if [[ -n "${1:-}" ]]; then
  SALLOC_ARGS+=(--time="$1")
fi

exec salloc "${SALLOC_ARGS[@]}" srun --pty bash

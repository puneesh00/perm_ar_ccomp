#!/bin/bash
# Launch a long-running interactive GPU session via salloc + srun --pty bash.
#
# Usage:
#   bash scripts/slurm/interactive_gpu.sh                # no time limit (partition default)
#   bash scripts/slurm/interactive_gpu.sh 5-00:00:00      # cap at 3 days
#
# Adjust cpus/mem/gres below to taste before running.

set -euo pipefail

SALLOC_ARGS=(
  --job-name=interactive_gpu
  --account=research
  --cpus-per-task=4
  --mem=64G
  --gres=gpu:1
)

if [[ -n "${1:-}" ]]; then
  SALLOC_ARGS+=(--time="$1")
fi

exec salloc "${SALLOC_ARGS[@]}" srun --pty bash

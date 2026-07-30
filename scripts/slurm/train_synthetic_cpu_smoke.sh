#!/bin/bash
#SBATCH --job-name=tabcomp_cpu_smoke
#SBATCH --output=logs/tabcomp_cpu_smoke_%j.out
#SBATCH --error=logs/tabcomp_cpu_smoke_%j.err
#SBATCH --account=research
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

echo "=============================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "=============================="

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export WANDB_MODE=offline

echo "Python: $(which python)"
python --version

python - <<'PY'
import numpy, torch
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
PY

echo "Starting smoke test training..."

python scripts/train_synthetic.py \
  --sampler mixture \
  --factorization parallel \
  --n-rows 32 --n-cols 6 \
  --n-context 8 --n-query 8 --n-episode-rows 16 \
  --steps 5 --eval-every 5 --eval-tasks 2 --log-every 1 \
  --d-model 16 --num-row-layers 1 --num-row-context-layers 1 --n-heads 2 \
  --device cpu \
  --out-dir results/cpu_smoke --run-name smoke_${SLURM_JOB_ID}

echo "Done: $(date)"

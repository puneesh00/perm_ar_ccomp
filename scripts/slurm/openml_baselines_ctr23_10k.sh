#!/bin/bash
#SBATCH --job-name=openml_ctr23_10k
#SBATCH --output=logs/openml_ctr23_10k_%j.out
#SBATCH --error=logs/openml_ctr23_10k_%j.err
#SBATCH --account=research
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

echo "Python: $(which python)"
python --version

echo "Starting ctr23 full-suite run (35 tasks, 10k row cap)..."

python scripts/run_openml_baselines.py \
  --suite ctr23 \
  --models rf,hgb,xgb \
  --max-train-rows 10000 \
  --max-test-rows 10000 \
  --n-jobs 8 \
  --run-name ctr23_rf_hgb_xgb_10k_${SLURM_JOB_ID}

echo "Done: $(date)"

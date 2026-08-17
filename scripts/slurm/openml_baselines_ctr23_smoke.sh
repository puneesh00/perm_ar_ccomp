#!/bin/bash
#SBATCH --job-name=openml_ctr23_smoke
#SBATCH --output=logs/openml_ctr23_smoke_%j.out
#SBATCH --error=logs/openml_ctr23_smoke_%j.err
#SBATCH --account=research
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

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

# Unbuffered stdout so progress shows up in the log in real time instead of
# being lost if the job gets killed (e.g. hitting the time limit) before
# Python's block-buffered stdout is flushed.
export PYTHONUNBUFFERED=1

# Pin BLAS/OpenMP thread pools to the allocated CPU count. Without this,
# sklearn/xgboost can each try to spawn threads based on the node's full
# core count (this cluster's nodes are shared and can be busy with other
# users' jobs), causing severe oversubscription/contention on a node we
# only asked for 8 cores on.
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

echo "Python: $(which python)"
python --version

python - <<'PY'
import sklearn, pandas, numpy, xgboost, lightgbm, catboost, openml
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("lightgbm:", lightgbm.__version__)
print("catboost:", catboost.__version__)
print("openml:", openml.__version__)
PY

# Downloads suite/task metadata and datasets from openml.org, so this node
# needs outbound internet access.
echo "Starting ctr23 smoke test..."

python scripts/run_openml_baselines.py \
  --suite ctr23 \
  --models rf,hgb,xgb \
  --max-tasks 3 \
  --max-train-rows 5000 \
  --max-test-rows 5000 \
  --n-jobs 8 \
  --run-name ctr23_smoke_${SLURM_JOB_ID}

echo "Done: $(date)"

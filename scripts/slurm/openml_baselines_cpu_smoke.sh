#!/bin/bash
#SBATCH --job-name=openml_baselines_smoke
#SBATCH --output=logs/openml_baselines_smoke_%j.out
#SBATCH --error=logs/openml_baselines_smoke_%j.err
#SBATCH --account=research
#SBATCH --time=00:30:00
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
echo "Starting cc18 smoke test..."

python scripts/run_openml_baselines.py \
  --suite cc18 \
  --models rf,hgb,xgb \
  --max-tasks 3 \
  --max-train-rows 5000 \
  --max-test-rows 5000 \
  --n-jobs 8 \
  --run-name cc18_smoke_${SLURM_JOB_ID}

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

#!/bin/bash
#SBATCH --job-name=tabcomp_overfit
#SBATCH --output=logs/tabcomp_overfit_%j.out
#SBATCH --error=logs/tabcomp_overfit_%j.err
#SBATCH --account=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

echo "=============================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "=============================="

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export WANDB_MODE=offline

echo "Python: $(which python)"
python --version

echo "Checking GPU..."
nvidia-smi || true

python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA is not available inside this job.")
PY

echo "Starting overfit-the-table training..."

# Small fixed table + a decent-sized model: n-context + n-query covers the
# whole table each episode, so this checks whether the model can memorize it
# (train/eval loss -> ~0) rather than testing generalization.
python scripts/train_synthetic.py \
  --sampler target \
  --factorization parallel \
  --n-rows 256 \
  --n-cols 16 \
  --n-context 128 \
  --n-query 128 \
  --steps 3000 \
  --eval-every 300 \
  --log-every 20 \
  --d-model 256 \
  --num-row-layers 4 \
  --num-row-context-layers 2 \
  --n-heads 8 \
  --device cuda \
  --out-dir results/overfit_table \
  --run-name target_parallel_gpu_overfit_${SLURM_JOB_ID}

echo "Done: $(date)"
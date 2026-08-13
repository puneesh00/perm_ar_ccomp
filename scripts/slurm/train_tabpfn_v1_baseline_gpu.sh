#!/bin/bash
#SBATCH --job-name=tabpfn_v1_baseline
#SBATCH --output=logs/tabpfn_v1_baseline_%j.out
#SBATCH --error=logs/tabpfn_v1_baseline_%j.err
#SBATCH --account=research
#SBATCH --time=48:00:00
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

echo "Starting TabPFN-v1-style reference baseline training..."
echo ""
echo "Matched to the target_singlestream_scmcomplex_d256_l8_ctx512_cols64_1M(_v2)"
echo "run: 512 context rows / 1 query row / 64 cols per table, paper-complexity"
echo "SCM prior (layers TNLU up to mean 6, hidden width TNLU up to mean 130,"
echo "no hard depth cap), d_model=256 / 8 layers / mlp_hidden=1024 (4x, matching"
echo "our convention), batch_tasks=8, 125000 steps -> ~1,000,000 tables seen."
echo ""
echo "Matches the per-table sampled class count (2-10, binary-heavy) scheme"
echo "used by the target run: TabPFNV1Model's output head is a fixed-width"
echo "nn.Linear(mlp_hidden, max_num_classes) classifier, and labels are"
echo "densified per-episode to their rank among the unique values observed"
echo "in that episode's context rows (see build_xy / _flatten_multiclass_targets"
echo "in train_tabpfn_v1_baseline.py), same mechanism real TabPFN uses. Output"
echo "slots beyond an episode's realized class count are masked to -inf."
echo ""
echo "Running with --amp-dtype bf16. This model's datapoint_attn is plain"
echo "nn.MultiheadAttention over N=~513 rows, folded batch*columns deep"
echo "(B*65 independent attention instances per layer at these settings) --"
echo "each materializes an [N,N] attention matrix, which adds up fast across"
echo "8 layers even though there's only one token per row (not per cell like"
echo "our axial model). fp32 OOM'd on a 40GB card even at batch_tasks=4;"
echo "bf16 roughly halves activation memory and was needed to fit here."
echo ""

python scripts/train_tabpfn_v1_baseline.py \
  --tabpfn-prior-type scm \
  --tabpfn-layers-mu-max 6.0 \
  --tabpfn-layers-max 1000000000 \
  --tabpfn-hidden-mu-max 130.0 \
  --fresh-n-rows 513 \
  --n-cols 64 \
  --n-context 512 \
  --n-query 1 \
  --eval-n-context 512 \
  --eval-n-query 1 \
  --eval-tasks 30 \
  --eval-every 2500 \
  --steps 125000 \
  --batch-tasks 8 \
  --log-every 100 \
  --checkpoint-every 25000 \
  --d-model 256 \
  --n-heads 4 \
  --mlp-hidden 1024 \
  --n-layers 8 \
  --max-num-classes 10 \
  --warmup-steps 1000 \
  --lr-min-ratio 0.1 \
  --amp-dtype bf16 \
  --out-dir results/synthetic_v2 \
  --run-name tabpfn_v1_baseline_scmcomplex_d256_l8_ctx512_cols64_1M_${SLURM_JOB_ID}

echo "Done: $(date)"

#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu-test-%j.out
#SBATCH --error=logs/gpu-test-%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

nvidia-smi

/usr/bin/time -v python - <<'PY'
import time
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

# Allocate matrices once.
size = 8192
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Warm up the GPU.
for _ in range(3):
    c = torch.matmul(a, b)

torch.cuda.synchronize()

# Measure GPU execution.
start = time.perf_counter()

for _ in range(10):
    c = torch.matmul(a, b)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"GPU workload elapsed time: {elapsed:.3f} seconds")
PY

echo "End: $(date)"

#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs results slurm_state

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python3 scripts/slurm_model_dispatch.py

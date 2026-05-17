#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${JANUS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/llm_clients.py" ]]; then
  PROJECT_DIR="/mnt/iusers01/nactem01/e81040pg/codes/Janus"
fi
cd "$PROJECT_DIR"

module purge
module load libs/cuda

source ~/.bashrc
conda activate reasoning

JANUS_ENV_FILE="${JANUS_ENV_FILE:-$PROJECT_DIR/.env}"
if [[ -f "$JANUS_ENV_FILE" ]]; then
  set -a
  source "$JANUS_ENV_FILE"
  set +a
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python3 scripts/slurm_model_dispatch.py

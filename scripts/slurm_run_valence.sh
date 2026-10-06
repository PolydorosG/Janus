#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${JANUS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/experiments/classify_fact_valence.py" ]]; then
  PROJECT_DIR="/mnt/iusers01/nactem01/e81040pg/codes/Janus"
fi
cd "$PROJECT_DIR"

module purge
module load libs/cuda
source "$HOME/.bashrc"
conda activate "${CONDA_ENV:-reasoning}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
LIMIT="${LIMIT:-}"
QWEN_THINKING="${QWEN_THINKING:-0}"

command=(
  python3 experiments/classify_fact_valence.py
  --model "$MODEL"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --qwen-thinking "$QWEN_THINKING"
)
if [[ -n "$LIMIT" ]]; then
  command+=(--limit "$LIMIT")
fi
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  command+=(--load-in-4bit)
fi
"${command[@]}"


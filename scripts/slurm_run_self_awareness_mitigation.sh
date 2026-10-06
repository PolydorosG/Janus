#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${JANUS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/experiments/run_self_awareness_mitigation.py" ]]; then
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
PROVIDER="${PROVIDER:-qwen_local}"
NEUTRAL_BASELINE="${NEUTRAL_BASELINE:-results/qwen3_8b_no_thinking.jsonl}"
OUTPUT="${OUTPUT:-results/qwen3_8b_no_thinking_self_awareness.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
MAX_TOKENS="${MAX_TOKENS:-768}"
LIMIT="${LIMIT:-}"
QWEN_THINKING="${QWEN_THINKING:-0}"

command=(
  python3 experiments/run_self_awareness_mitigation.py
  --provider "$PROVIDER"
  --model "$MODEL"
  --neutral-baseline "$NEUTRAL_BASELINE"
  --output "$OUTPUT"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --max-tokens "$MAX_TOKENS"
  --qwen-thinking "$QWEN_THINKING"
)
if [[ -n "$LIMIT" ]]; then
  command+=(--limit "$LIMIT")
fi
"${command[@]}"

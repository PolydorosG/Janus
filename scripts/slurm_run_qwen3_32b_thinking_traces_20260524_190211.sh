#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -J qwen32b_think_traces
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

OUTPUT_PATH="results/qwen3_32b_thinking_reasoning_traces_20260524_190211.jsonl"

python3 llm_clients.py \
  --provider qwen_local \
  --model Qwen/Qwen3-32B \
  --output "$OUTPUT_PATH" \
  --seed "${SEED:-1234}" \
  --device-map "${DEVICE_MAP:-auto}" \
  --dtype "${DTYPE:-bfloat16}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --max-tokens "${MAX_TOKENS:-2048}" \
  --qwen-thinking 1 \
  --fail-on-error

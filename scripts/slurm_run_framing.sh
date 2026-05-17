#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${JANUS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/judge_framing.py" ]]; then
  PROJECT_DIR="/mnt/iusers01/nactem01/e81040pg/codes/Janus"
fi
cd "$PROJECT_DIR"

if command -v module >/dev/null 2>&1; then
  module purge
  module load libs/cuda
fi

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV:-reasoning}"
fi

JANUS_ENV_FILE="${JANUS_ENV_FILE:-$PROJECT_DIR/.env}"
if [[ -f "$JANUS_ENV_FILE" ]]; then
  set -a
  source "$JANUS_ENV_FILE"
  set +a
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-auto}"
QWEN_THINKING="${QWEN_THINKING:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-}"
NO_RESUME="${NO_RESUME:-0}"
DATASET="${DATASET:-data/janus.csv}"

if [[ -n "${MATCHED_FILES:-}" ]]; then
  read -r -a matched_files <<< "$MATCHED_FILES"
else
  mapfile -t matched_files < <(
    find results -maxdepth 1 -type f -name '*_matched.jsonl' | sort
  )
fi

if [[ "${#matched_files[@]}" -eq 0 ]]; then
  echo "No matched JSONL files found."
  exit 0
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#matched_files[@]} )); then
    echo "Array task index $SLURM_ARRAY_TASK_ID is outside 0..$((${#matched_files[@]} - 1))."
    exit 1
  fi
  matched_files=("${matched_files[$SLURM_ARRAY_TASK_ID]}")
fi

echo "Running framing evaluation with judge model: $JUDGE_MODEL"
echo "Found ${#matched_files[@]} matched file(s)."
echo "Batch size: $BATCH_SIZE"

for matched_path in "${matched_files[@]}"; do
  if [[ ! -f "$matched_path" ]]; then
    echo "Skipping missing matched input: $matched_path"
    continue
  fi

  raw_path="${matched_path%_matched.jsonl}.jsonl"
  output_path="${matched_path%_matched.jsonl}_framing.jsonl"
  if [[ ! -f "$raw_path" ]]; then
    echo "Skipping $matched_path because raw result file is missing: $raw_path"
    continue
  fi

  command=(
    python3 judge_framing.py "$raw_path" "$matched_path"
    --output "$output_path"
    --model "$JUDGE_MODEL"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --device-map "$DEVICE_MAP"
    --dtype "$DTYPE"
    --qwen-thinking "$QWEN_THINKING"
    --batch-size "$BATCH_SIZE"
    --dataset "$DATASET"
  )

  if [[ "$TRUST_REMOTE_CODE" == "1" || "$TRUST_REMOTE_CODE" == "true" ]]; then
    command+=(--trust-remote-code)
  fi
  if [[ -n "$LIMIT" ]]; then
    command+=(--limit "$LIMIT")
  fi
  if [[ "$NO_RESUME" == "1" || "$NO_RESUME" == "true" ]]; then
    command+=(--no-resume)
  fi

  echo "Evaluating framing $matched_path -> $output_path"
  "${command[@]}"
done

echo "Framing evaluation complete."

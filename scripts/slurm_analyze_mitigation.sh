#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${JANUS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/experiments/analyze_mitigation.py" ]]; then
  PROJECT_DIR="/mnt/iusers01/nactem01/e81040pg/codes/Janus"
fi
cd "$PROJECT_DIR"

python3 experiments/analyze_mitigation.py \
  --run Qwen3-8B \
    metrics/qwen3_8b_no_thinking_self_awareness_direct/core_metrics.csv \
    metrics/qwen3_8b_no_thinking_self_awareness/core_metrics.csv \
  --run Llama-3.1-8B \
    metrics/llama_3_1_8b_instruct_self_awareness_direct/core_metrics.csv \
    metrics/llama_3_1_8b_instruct_self_awareness/core_metrics.csv

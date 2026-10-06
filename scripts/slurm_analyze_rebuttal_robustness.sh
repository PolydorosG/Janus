#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${JANUS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/experiments/analyze_framing_judges.py" ]]; then
  PROJECT_DIR="/mnt/iusers01/nactem01/e81040pg/codes/Janus"
fi
cd "$PROJECT_DIR"

python3 experiments/analyze_framing_judges.py \
  --alternate-dir "${ALTERNATE_FRAMING_DIR:-analysis/framing_llama70}" \
  --alternate-tag "${ALTERNATE_FRAMING_TAG:-framing_llama70}" \
  --output-dir "${FRAMING_OUTPUT_DIR:-analysis/judge_robustness_llama70}"

python3 experiments/analyze_valence_sensitivity.py \
  "${VALENCE_JSONL:-analysis/valence/meta_llama_llama_3_1_70b_instruct.jsonl}" \
  --output-dir "${VALENCE_OUTPUT_DIR:-analysis/valence_sensitivity_llama70}"

echo "Rebuttal robustness analyses complete."

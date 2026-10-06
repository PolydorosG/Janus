#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${JANUS_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/metrics_pipeline.py" ]]; then
  PROJECT_DIR="/mnt/iusers01/nactem01/e81040pg/codes/Janus"
fi
cd "$PROJECT_DIR"

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV:-reasoning}"
fi

: "${RAW:?Set RAW to the three-condition raw JSONL file}"
: "${MATCHED:?Set MATCHED to the sentence-matched JSONL file}"
: "${FRAMING:?Set FRAMING to the framing-judgment JSONL file}"

STEM="$(basename "${RAW%.jsonl}")"
MITIGATED_OUTPUT_DIR="${MITIGATED_OUTPUT_DIR:-metrics/$STEM}"
DIRECT_OUTPUT_DIR="${DIRECT_OUTPUT_DIR:-metrics/${STEM}_direct}"

python3 metrics_pipeline.py "$RAW" "$MATCHED" \
  --framing-jsonl "$FRAMING" \
  --output-dir "$MITIGATED_OUTPUT_DIR" \
  --target-condition latent_goal_self_awareness \
  --test-sample-id ""

python3 metrics_pipeline.py "$RAW" "$MATCHED" \
  --framing-jsonl "$FRAMING" \
  --output-dir "$DIRECT_OUTPUT_DIR" \
  --reference-condition latent_goal \
  --target-condition latent_goal_self_awareness \
  --test-sample-id ""

echo "Mitigation metrics complete: $MITIGATED_OUTPUT_DIR and $DIRECT_OUTPUT_DIR"

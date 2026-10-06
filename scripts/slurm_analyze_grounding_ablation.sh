#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
cd /mnt/iusers01/nactem01/e81040pg/codes/Janus
source "$HOME/.bashrc"
conda activate reasoning
export PYTHONUNBUFFERED=1
for model_key in qwen3_8b_no_thinking llama_3_1_8b_instruct; do
  raw="results/arr_grounding_ablation_2026_09/$model_key.jsonl"
  python experiments/analyze_grounding_ablation.py --validate frame --model-key "$model_key"
  python metrics_pipeline.py "$raw" "${raw%.jsonl}_matched.jsonl" \
    --framing-jsonl "${raw%.jsonl}_framing.jsonl" \
    --output-dir "metrics/arr_grounding_ablation_2026_09/$model_key" \
    --reference-condition neutral_no_grounding --target-condition latent_goal --test-sample-id ""
done
python experiments/analyze_grounding_ablation.py

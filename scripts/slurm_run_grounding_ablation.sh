#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 08:00:00
#SBATCH --array=0-1
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
set -euo pipefail
cd /mnt/iusers01/nactem01/e81040pg/codes/Janus
module purge
module load libs/cuda
source "$HOME/.bashrc"
conda activate reasoning
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
models=(qwen3_8b_no_thinking llama_3_1_8b_instruct)
model_key="${models[${SLURM_ARRAY_TASK_ID:?}]}"
stage="${1:?Specify generate, match, or frame}"
raw="results/arr_grounding_ablation_2026_09/$model_key.jsonl"
matched="${raw%.jsonl}_matched.jsonl"
framing="${raw%.jsonl}_framing.jsonl"
judge_snapshot=$(python -c 'import json; print(json.load(open("results/arr_grounding_ablation_2026_09/qwen3_8b_no_thinking.manifest.json"))["settings"]["snapshot"])')
case "$stage" in
  generate)
    python experiments/run_grounding_ablation.py --model-key "$model_key"
    ;;
  match)
    python judge_fact_units.py "$raw" --output "$matched" --model "$judge_snapshot" --dtype bfloat16 --qwen-thinking 0 --batch-size 4 --max-new-tokens 2048
    ;;
  frame)
    python judge_framing.py "$raw" "$matched" --output "$framing" --model "$judge_snapshot" --dtype bfloat16 --qwen-thinking 0 --batch-size 8 --max-new-tokens 512
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
python experiments/analyze_grounding_ablation.py --validate "$stage" --model-key "$model_key"

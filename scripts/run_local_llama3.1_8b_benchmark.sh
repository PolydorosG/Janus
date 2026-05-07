#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LIMIT="${LIMIT:-160}"
MAX_TOKENS="${MAX_TOKENS:-768}"
SEED="${SEED:-1234}"
OUTPUT="${OUTPUT:-results/llama_3_1_8b_instruct.jsonl}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-4}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

python3 -c "import torch, transformers"

python3 llm_clients.py \
  --provider llama_local \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --max-tokens "$MAX_TOKENS" \
  --seed "$SEED" \
  --output "$OUTPUT" \
  --device-map "$DEVICE_MAP" \
  --dtype "$DTYPE" \
  --batch-size "$BATCH_SIZE"

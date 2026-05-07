#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-8B}"
LIMIT="${LIMIT:-160}"
MAX_TOKENS="${MAX_TOKENS:-768}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SEED="${SEED:-1234}"
OUTPUT="${OUTPUT:-results/qwen_3_8b.jsonl}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-auto}"
QWEN_THINKING="${QWEN_THINKING:-0}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

python3 -c "import torch, transformers; from packaging.version import parse; import sys; sys.exit('Qwen3 requires transformers>=4.51.0') if parse(transformers.__version__) < parse('4.51.0') else None"

python3 llm_clients.py \
  --provider qwen_local \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --output "$OUTPUT" \
  --device-map "$DEVICE_MAP" \
  --dtype "$DTYPE" \
  --qwen-thinking "$QWEN_THINKING"

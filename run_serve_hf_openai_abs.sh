#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="/weka/oe-training-default/reza/zero/evoskill-march11/fara/src/fara/serve_hf_openai.py"
PYTHONPATH_ROOT="/weka/oe-training-default/reza/zero/evoskill-march11/fara/src"

# Hardcoded server/model settings.
MODEL_PATH="microsoft/Fara-7B"
MODEL_NAME="fara-hf-local"
HOST="0.0.0.0"
PORT="5000"
DTYPE="bfloat16"
DEVICE_MAP="auto"

export PYTHONPATH="${PYTHONPATH_ROOT}:${PYTHONPATH:-}"

exec "$PYTHON_BIN" "$SCRIPT_PATH" \
  --model-path "$MODEL_PATH" \
  --model-name "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP"

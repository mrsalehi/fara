#!/usr/bin/env bash
# Multi-GPU launcher for train_fara.py via `accelerate launch`.
# Usage:
#   bash scripts/train_fara.sh [extra args forwarded to train_fara.py]
#
# Single-GPU: `python scripts/train_fara.py ...` works fine too.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${NUM_GPUS:=$(python -c 'import torch; print(torch.cuda.device_count())')}"
: "${DATA_PATH:=/gpfs/projects/raivn/reza/MolmoWeb-SyntheticTrajs/data/from_template-00000.parquet}"
: "${OUTPUT_DIR:=$REPO_ROOT/results/fara_sft}"
: "${MODEL_ID:=microsoft/Fara-7B}"

accelerate launch \
  --num_processes "$NUM_GPUS" \
  --mixed_precision bf16 \
  "$SCRIPT_DIR/train_fara.py" \
    --data_path "$DATA_PATH" \
    --model_id "$MODEL_ID" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

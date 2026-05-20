#!/usr/bin/env bash
set -euo pipefail

# Run FARA on the full local WebVoyager task file and save trajectories.
#
# Override with environment variables when needed:
#   OUT_URL=/path/to/output/root
#   RUN_ID=my_run_id
#   EVAL_DATA_URL=/path/to/WebVoyager_data_08312025.jsonl
#   LOCAL_MODEL_ID=/path/or/hf/model
#   PROCESSES=1
#   MAX_ROUNDS=30
#   NO_MULTISCALE=1
#   MODEL_ENDPOINT=/path/to/endpoint_config.json  # uses hosted endpoint instead of local model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/raivn/reza/miniconda3/envs/fara_webeval/bin/python}"

OUT_URL="${OUT_URL:-/gpfs/scrubbed/reza/fara/results/fara_webvoyager_all}"
RUN_ID="${RUN_ID:-fara_webvoyager_all}"
EVAL_DATA_URL="${EVAL_DATA_URL:-${REPO_ROOT}/webeval/data/webvoyager/WebVoyager_data_08312025.jsonl}"
LOCAL_MODEL_ID="${LOCAL_MODEL_ID:-microsoft/Fara-7B}"
PROCESSES="${PROCESSES:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"
SEED="${SEED:-42}"

args=(
  "${REPO_ROOT}/webeval/scripts/webvoyager.py"
  --eval_data_url "${EVAL_DATA_URL}"
  --skip_eval
  --subsample 1.0
  --seed "${SEED}"
  --out_url "${OUT_URL}"
  --max_rounds "${MAX_ROUNDS}"
  --processes "${PROCESSES}"
  --run_id "${RUN_ID}"
)

if [[ -n "${MODEL_ENDPOINT:-}" ]]; then
  args+=(--model_endpoint "${MODEL_ENDPOINT}")
else
  args+=(
    --model_endpoint "${REPO_ROOT}/endpoint_configs/fara_hf_local.json"
    --local
    --local_model_id "${LOCAL_MODEL_ID}"
  )
fi

if [[ "${NO_MULTISCALE:-1}" == "1" ]]; then
  args+=(--no_multiscale)
fi

cd "${REPO_ROOT}/webeval/scripts"
echo "[run] output root: ${OUT_URL}"
echo "[run] run id: ${RUN_ID}"
echo "[run] eval data: ${EVAL_DATA_URL}"
exec "${PYTHON_BIN}" "${args[@]}"

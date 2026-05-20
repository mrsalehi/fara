#!/usr/bin/env bash
#SBATCH --job-name=fara-sft-2k
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --chdir=/gpfs/projects/raivn/reza/fara
#SBATCH --output=/gpfs/projects/raivn/reza/fara/logs/slurm-%x-%j.out
#SBATCH --error=/gpfs/projects/raivn/reza/fara/logs/slurm-%x-%j.out

: <<'USAGE'
Mirrors the FARA_TRAIN entry in .vscode/launch.json, scaled to multi-GPU
via accelerate launch. Defaults: LoRA + multiscale + full dataset (no cap).

Submit (default = LoRA + multiscale, full dataset):
    sbatch scripts/train_fara_2k.sbatch

-----Override hyperparameters / paths via env vars:
NUM_GPUS=8                                 # default: torch.cuda.device_count()
DATA_PATH=/path/to/file.parquet
MODEL_ID=microsoft/Fara-7B
OUTPUT_DIR=/path/to/run                    # default: results/fara_sft_2k_<jobid>
MAX_SAMPLES=2000                           # cap dataset; unset = use all
VAL_SPLIT_RATIO=0.1                        # hold out 10% for validation
VAL_MAX_SAMPLES=500                        # optional cap on val set
EVAL_STEPS=200                             # how often to run eval (default 500)
INF_EVAL_SAMPLES=4                         # generations logged to wandb each eval (0/unset = off)
INF_EVAL_MAX_NEW_TOKENS=256                # max new tokens per inference-eval sample
INF_EVAL_SEED=0                            # seed for inference-eval sample selection
SAMPLING_STRATEGY=none                     # full retained trajectories; or decision_point/full_trajectory
MAX_TRAJECTORY_TURNS=16                    # keep only rows with <= this many trajectory steps
WANDB_RUN_NAME=fara-sft-2k-lora-singlescale
WANDB_TAGS=lora,singlescale,2k
EXTRA_ARGS="--no_multiscale"               # appended to train_fara.py CLI

-----Full data + 10% val example:
LORA=0 EPOCHS=3 VAL_SPLIT_RATIO=0.1 sbatch scripts/train_fara_2k.sbatch

-----Single-scale ablation example:
EXTRA_ARGS="--no_multiscale" \
WANDB_RUN_NAME=fara-sft-2k-lora-singlescale \
WANDB_TAGS=lora,singlescale,2k \
OUTPUT_DIR=/gpfs/projects/raivn/reza/fara/results/fara_sft_2k_singlescale \
sbatch scripts/train_fara_2k.sbatch

-----Domain presets (default = full WebVoyager set; override via ALLOWED_DOMAINS):
# arxiv + allrecipes only:
ALLOWED_DOMAINS="arxiv.org,allrecipes.com" sbatch scripts/train_fara_2k.sbatch

# arxiv only:
ALLOWED_DOMAINS="arxiv.org" sbatch scripts/train_fara_2k.sbatch

# full WebVoyager set (explicit form — same as default):
ALLOWED_DOMAINS="google.com,wolframalpha.com,booking.com,dictionary.cambridge.org,arxiv.org,github.com,allrecipes.com,coursera.org,amazon.com,espn.com,huggingface.co,apple.com,bbc.com" sbatch scripts/train_fara_2k.sbatch

-----FSDP full fine-tune (shards params + grads + optimizer across GPUs):
LORA=0 \
EXTRA_ARGS="--fsdp full_shard,auto_wrap --fsdp_config /gpfs/projects/raivn/reza/fara/scripts/fsdp.json" \
WANDB_TAGS=fullft,fsdp,2k \
sbatch scripts/train_fara_2k.sbatch
USAGE

set -euo pipefail

source /gpfs/projects/raivn/reza/miniconda3/etc/profile.d/conda.sh
conda activate fara_webeval
REPO_ROOT=/gpfs/projects/raivn/reza/fara/

: "${NUM_GPUS:=$(python -c 'import torch; print(torch.cuda.device_count())')}"
: "${DATA_PATH:=/gpfs/scrubbed/reza/MolmoWeb-SyntheticTrajs/data_filtered/}"
: "${MODEL_ID:=microsoft/Fara-7B}"
: "${RESULTS_ROOT:=/gpfs/scrubbed/reza/fara/results}"

# LoRA toggle: LORA=1 (default) for LoRA fine-tune, LORA=0 for full FT.
LORA_FLAG=""
_FT_TAG="fullft"
if [[ "${LORA:-1}" == "1" ]]; then
  LORA_FLAG="--lora"
  _FT_TAG="lora"
fi

# Scale tag: add "singlescale" to run name if --no_multiscale is in EXTRA_ARGS
_SCALE_TAG=""
if [[ "${EXTRA_ARGS:-}" == *"--no_multiscale"* ]]; then
  _SCALE_TAG="-singlescale"
fi
: "${WANDB_RUN_NAME:=fara-sft-2k-${_FT_TAG}${_SCALE_TAG}-${SLURM_JOB_ID:-local}}"

# OUTPUT_DIR mirrors WANDB_RUN_NAME so the run dir on disk matches the W&B run.
: "${OUTPUT_DIR:=$RESULTS_ROOT/$WANDB_RUN_NAME}"

# Optional dataset cap: omit --max_samples entirely when MAX_SAMPLES is unset/empty
# so train_fara.py uses the full dataset.
MAX_SAMPLES_FLAG=""
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_FLAG="--max_samples ${MAX_SAMPLES}"
fi

# Optional held-out val split (e.g., VAL_SPLIT_RATIO=0.1 -> 10% val).
VAL_FLAGS=""
if [[ -n "${VAL_SPLIT_RATIO:-}" ]]; then
  VAL_FLAGS="--val_split_ratio ${VAL_SPLIT_RATIO}"
  [[ -n "${VAL_MAX_SAMPLES:-}" ]] && VAL_FLAGS+=" --val_max_samples ${VAL_MAX_SAMPLES}"
  VAL_FLAGS+=" --eval_steps ${EVAL_STEPS:-128}"
fi

# Optional inference-eval viz (logs generations to wandb under eval/inf/samples).
# Only enabled when a val split exists; mirrors FARA_TRAIN_ACCELERATE in launch.json.
INF_EVAL_FLAGS=""
if [[ -n "${VAL_SPLIT_RATIO:-}" && -n "${INF_EVAL_SAMPLES:-}" && "${INF_EVAL_SAMPLES}" != "0" ]]; then
  INF_EVAL_FLAGS="--inf_eval_samples ${INF_EVAL_SAMPLES}"
  INF_EVAL_FLAGS+=" --inf_eval_max_new_tokens ${INF_EVAL_MAX_NEW_TOKENS:-256}"
  INF_EVAL_FLAGS+=" --inf_eval_seed ${INF_EVAL_SEED:-0}"
fi

# Optional vision-encoder LR override. VISION_LEARNING_RATE wins over VISION_LR_RATIO.
VISION_LR_FLAGS=""
[[ -n "${VISION_LEARNING_RATE:-}" ]] && VISION_LR_FLAGS+=" --vision_learning_rate ${VISION_LEARNING_RATE}"
[[ -n "${VISION_LR_RATIO:-}" ]]      && VISION_LR_FLAGS+=" --vision_lr_ratio ${VISION_LR_RATIO}"

# Auto-derive W&B tags from active flags / sample count.
_TAGS=("$_FT_TAG")
case "${EXTRA_ARGS:-}" in *--no_multiscale*) _TAGS+=("singlescale") ;; *) _TAGS+=("multiscale") ;; esac
_TAGS+=("${NUM_GPUS:-auto}gpu")
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  if   (( MAX_SAMPLES >= 1000 )); then _TAGS+=("$(( MAX_SAMPLES / 1000 ))k")
  else                                  _TAGS+=("${MAX_SAMPLES}")
  fi
else
  _TAGS+=("full")
fi
_TAGS+=("ep${EPOCHS:-1}")
_TAGS+=("lr${LEARNING_RATE:-1e-5}")
_TAGS+=("${SAMPLING_STRATEGY:-none}")
_TAGS+=("max${MAX_TRAJECTORY_TURNS:-16}turns")
[[ -n "${VAL_SPLIT_RATIO:-}" ]] && _TAGS+=("val${VAL_SPLIT_RATIO}")

WANDB_TAGS_AUTO=$(IFS=,; echo "${_TAGS[*]}")
WANDB_TAGS="${WANDB_TAGS:-$WANDB_TAGS_AUTO}"

export HF_TOKEN="${HF_TOKEN:-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}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/gpfs/scrubbed/reza/hf_cache
export HF_HOME=/gpfs/scrubbed/reza/hf_cache
export HF_HUB_CACHE=/gpfs/scrubbed/reza/hf_cache/hub

# Pin a wandb run id so the post-hoc inference-eval step can resume the
# same run and log per-checkpoint HTMLs on the original training timeline.
export WANDB_RUN_ID="${WANDB_RUN_ID:-${SLURM_JOB_ID:-local}}"

# Directory for debug dumps when multiscale kept_patches is empty; default per-run
export FARA_EMPTY_KEPT_DUMP_DIR="${FARA_EMPTY_KEPT_DUMP_DIR:-$RESULTS_ROOT/fara_empty_kept_${SLURM_JOB_ID:-local}}"

echo "[$(date)] starting on $(hostname); GPUs=$NUM_GPUS; output=$OUTPUT_DIR"

BASE_PORT=29500
OFFSET=$(( RANDOM % 1000 ))  # random offset between 0 and 999
export MASTER_PORT=$(( BASE_PORT + OFFSET ))

accelerate launch \
  --num_processes "$NUM_GPUS" \
  --mixed_precision bf16 \
  "/gpfs/projects/raivn/reza/fara/scripts/train_fara.py" \
    --data_path "$DATA_PATH" \
    --model_id "$MODEL_ID" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_batch_size ${PER_DEVICE_BATCH_SIZE:-1} \
    --gradient_accumulation_steps 8 \
    --learning_rate ${LEARNING_RATE:-1e-5} \
    --num_epochs ${EPOCHS:-1} \
    --max_seq_length ${MAX_SEQ_LENGTH:-16384} \
    --logging_steps 1 \
    --save_steps 128 \
    --save_total_limit ${SAVE_TOTAL_LIMIT:-3} \
    --max_trajectory_turns ${MAX_TRAJECTORY_TURNS:-8} \
    ${MAX_SAMPLES_FLAG} \
    ${VAL_FLAGS} \
    ${INF_EVAL_FLAGS} \
    ${VISION_LR_FLAGS} \
    --sampling_strategy ${SAMPLING_STRATEGY:-none} \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --allowed_domains "${ALLOWED_DOMAINS:-google.com,wolframalpha.com,booking.com,dictionary.cambridge.org,arxiv.org,github.com,allrecipes.com,coursera.org,amazon.com,espn.com,huggingface.co,apple.com,bbc.com}" \
    ${LORA_FLAG} \
    --report_to wandb \
    --domain_filter_mode ${DOMAIN_FILTER_MODE:-strict} \
    --wandb_entity mrsalehi \
    --wandb_project fara-multiscale \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_tags "$WANDB_TAGS" \
    ${EXTRA_ARGS:-}

echo "[$(date)] training finished"

# ----- Post-training: single-GPU inference eval over saved checkpoints. -----
# Loads each checkpoint-* on cuda:0 (no FSDP), runs InferenceEvalCallback's
# rendering pass on the same val split, and logs HTMLs to the same wandb run
# (resumed via WANDB_RUN_ID) plotted against eval/inf/ckpt_step.
if [[ -n "${VAL_SPLIT_RATIO:-}" && -n "${INF_EVAL_SAMPLES:-}" && "${INF_EVAL_SAMPLES}" != "0" ]]; then
  echo "[$(date)] starting post-hoc inference eval on cuda:0"

  POSTHOC_MULTISCALE_FLAG=""
  if [[ "${EXTRA_ARGS:-}" == *"--no_multiscale"* ]]; then
    POSTHOC_MULTISCALE_FLAG="--no_multiscale"
  fi

  CUDA_VISIBLE_DEVICES=0 python /gpfs/projects/raivn/reza/fara/scripts/inference_eval_checkpoints.py \
      --ckpt_dir "$OUTPUT_DIR" \
      --base_model_id "$MODEL_ID" \
      --gpu_id 0 \
      --data_path "$DATA_PATH" \
      --allowed_domains "${ALLOWED_DOMAINS:-google.com,wolframalpha.com,booking.com,dictionary.cambridge.org,arxiv.org,github.com,allrecipes.com,coursera.org,amazon.com,espn.com,huggingface.co,apple.com,bbc.com}" \
      --domain_filter_mode "${DOMAIN_FILTER_MODE:-strict}" \
      --sampling_strategy "${SAMPLING_STRATEGY:-none}" \
      --max_seq_length "${MAX_SEQ_LENGTH:-16384}" \
      --max_trajectory_turns "${MAX_TRAJECTORY_TURNS:-8}" \
      ${MAX_SAMPLES_FLAG} \
      --val_split_ratio "${VAL_SPLIT_RATIO}" \
      --val_split_seed 42 \
      ${VAL_MAX_SAMPLES:+--val_max_samples "${VAL_MAX_SAMPLES}"} \
      ${POSTHOC_MULTISCALE_FLAG} \
      --bf16 --attn_implementation flash_attention_2 \
      --inf_eval_samples "${INF_EVAL_SAMPLES}" \
      --inf_eval_seed "${INF_EVAL_SEED:-0}" \
      --inf_eval_max_new_tokens "${INF_EVAL_MAX_NEW_TOKENS:-256}" \
      --wandb_entity mrsalehi \
      --wandb_project fara-multiscale \
      --wandb_run_id "$WANDB_RUN_ID"
fi

echo "[$(date)] done"

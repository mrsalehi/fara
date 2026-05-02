#!/usr/bin/env bash
#SBATCH --job-name=fara-sft-2k
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --chdir=/gpfs/projects/raivn/reza/fara
#SBATCH --output=/gpfs/projects/raivn/reza/fara/logs/slurm-%x-%j.out
#SBATCH --error=/gpfs/projects/raivn/reza/fara/logs/slurm-%x-%j.out

: <<'USAGE'
Mirrors the FARA_TRAIN entry in .vscode/launch.json, scaled to multi-GPU
via accelerate launch. Defaults: LoRA + multiscale + 2000-sample cap.

Submit (default = LoRA + multiscale):
    sbatch scripts/train_fara_2k.sbatch

-----Override hyperparameters / paths via env vars:
NUM_GPUS=8                                 # default: torch.cuda.device_count()
DATA_PATH=/path/to/file.parquet
MODEL_ID=microsoft/Fara-7B
OUTPUT_DIR=/path/to/run                    # default: results/fara_sft_2k_<jobid>
WANDB_RUN_NAME=fara-sft-2k-lora-singlescale
WANDB_TAGS=lora,singlescale,2k
EXTRA_ARGS="--no_multiscale"               # appended to train_fara.py CLI

-----Single-scale ablation example:
EXTRA_ARGS="--no_multiscale" \
WANDB_RUN_NAME=fara-sft-2k-lora-singlescale \
WANDB_TAGS=lora,singlescale,2k \
OUTPUT_DIR=/gpfs/projects/raivn/reza/fara/results/fara_sft_2k_singlescale \
sbatch scripts/train_fara_2k.sbatch
USAGE

set -euo pipefail

source /gpfs/projects/raivn/reza/miniconda3/etc/profile.d/conda.sh
conda activate fara_webeval
REPO_ROOT=/gpfs/projects/raivn/reza/fara/

: "${NUM_GPUS:=$(python -c 'import torch; print(torch.cuda.device_count())')}"
: "${DATA_PATH:=/gpfs/scrubbed/reza/MolmoWeb-SyntheticTrajs/data/from_template-00000.parquet}"
: "${MODEL_ID:=microsoft/Fara-7B}"
: "${OUTPUT_DIR:=$REPO_ROOT/results/fara_sft_2k_${SLURM_JOB_ID:-local}}"
: "${WANDB_RUN_NAME:=fara-sft-2k-lora-${SLURM_JOB_ID:-local}}"

# LoRA toggle: LORA=1 (default) for LoRA fine-tune, LORA=0 for full FT.
LORA_FLAG=""
[[ "${LORA:-1}" == "1" ]] && LORA_FLAG="--lora"

# Auto-derive W&B tags from active flags / sample count.
_TAGS=()
[[ "${LORA:-1}" == "1" ]] && _TAGS+=("lora") || _TAGS+=("fullft")
case "${EXTRA_ARGS:-}" in *--no_multiscale*) _TAGS+=("singlescale") ;; *) _TAGS+=("multiscale") ;; esac
_MAX_SAMPLES="${MAX_SAMPLES:-2000}"
if   (( _MAX_SAMPLES >= 1000 )); then _TAGS+=("$(( _MAX_SAMPLES / 1000 ))k")
else                                   _TAGS+=("${_MAX_SAMPLES}")
fi
_TAGS+=("ep${EPOCHS:-1}")
WANDB_TAGS_AUTO=$(IFS=,; echo "${_TAGS[*]}")
WANDB_TAGS="${WANDB_TAGS:-$WANDB_TAGS_AUTO}"

export HF_TOKEN="${HF_TOKEN:-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}"
export TOKENIZERS_PARALLELISM=false

# Directory for debug dumps when multiscale kept_patches is empty; default per-run
export FARA_EMPTY_KEPT_DUMP_DIR="${FARA_EMPTY_KEPT_DUMP_DIR:-$REPO_ROOT/results/fara_empty_kept_${SLURM_JOB_ID:-local}}"

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
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_epochs ${EPOCHS:-1} \
    --max_seq_length 8192 \
    --logging_steps 1 \
    --save_steps 200 \
    --max_samples ${MAX_SAMPLES:-2000} \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation sdpa \
    --allowed_domains arxiv.org,allrecipes.com \
    ${LORA_FLAG} \
    --report_to wandb \
    --wandb_entity mrsalehi \
    --wandb_project fara-multiscale \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_tags "$WANDB_TAGS" \
    ${EXTRA_ARGS:-}

echo "[$(date)] done"

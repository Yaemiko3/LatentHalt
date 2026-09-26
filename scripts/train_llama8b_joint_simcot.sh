#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname -- "$PROJECT_ROOT")"
STORAGE_ROOT="${LATENTHALT_STORAGE_ROOT:-$PROJECT_ROOT}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/datasets}"

# Reuse the 1B joint curriculum logic, losses, resume handling, and disk
# checkpoint lifecycle, with 8B-specific defaults and FSDP.
export SFT_MODEL_PATH="${LLAMA8B_SFT_MODEL_PATH:-${SFT_MODEL_PATH:-$STORAGE_ROOT/outputs/sft-cot-llama8b}}"
export TOKENIZER_PATH="${LLAMA8B_TOKENIZER_PATH:-${TOKENIZER_PATH:-$SFT_MODEL_PATH}}"
export AUXILIARY_MODEL_PATH="${LLAMA8B_AUXILIARY_MODEL_PATH:-${AUXILIARY_MODEL_PATH:-$WORKSPACE_ROOT/models/Llama-3.1-8B-Instruct}}"
export TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/gsm8k-aug/data/train-00000-of-00001.parquet}"
export VALIDATION_FILE="${VALIDATION_FILE:-$DATA_ROOT/gsm8k-aug/data/validation-00000-of-00001.parquet}"
export THINK_REGION_FILE="${LLAMA8B_THINK_REGION_FILE:-${THINK_REGION_FILE:-$STORAGE_ROOT/results/analysis/think_hidden_geometry_llama8b/think_region.safetensors}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/outputs/latent-halt-llama8b}"
export LOG_ROOT="${LOG_ROOT:-$STORAGE_ROOT/logs}"
export RUN_ID="${RUN_ID:-train-joint-simcot-llama8b-fsdp-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ -n "${NUM_GPUS:-}" && "$NUM_GPUS" != "${#VISIBLE_GPU_IDS[@]}" ]]; then
  echo "NUM_GPUS=$NUM_GPUS does not match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
export NUM_GPUS="${#VISIBLE_GPU_IDS[@]}"

# Match coconut/train_llama8b_gsm8k_aug.sh: one latent token per reasoning step.
export C_THOUGHT="${C_THOUGHT:-1}"

# Keep three epochs per stage and stop after 20 training epochs.
export EPOCHS_PER_STAGE="${EPOCHS_PER_STAGE:-3}"
export NUM_EPOCHS="${NUM_EPOCHS:-20}"

# Default: 4 examples x 8 accumulation steps x 2 H200 GPUs = global batch 64.
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

AUXILIARY_GRADIENT_CHECKPOINTING="${AUXILIARY_GRADIENT_CHECKPOINTING:-1}"
case "$AUXILIARY_GRADIENT_CHECKPOINTING" in
  1|true|TRUE|yes|YES) auxiliary_checkpoint_arg="--auxiliary_gradient_checkpointing" ;;
  0|false|FALSE|no|NO) auxiliary_checkpoint_arg="--no-auxiliary_gradient_checkpointing" ;;
  *)
    echo "AUXILIARY_GRADIENT_CHECKPOINTING must be a boolean; got: $AUXILIARY_GRADIENT_CHECKPOINTING" >&2
    exit 2
    ;;
esac

# --fsdp selects full_shard + LlamaDecoderLayer auto wrapping, use_orig_params,
# and no forward/backward prefetch in src/train_llama1b_simcot.py.
# Keep full checkpoint files compatible with the existing evaluation watcher.
export FSDP_STATE_DICT_TYPE=FULL_STATE_DICT

echo "8B joint SIM-CoT: FSDP FULL_SHARD, BF16, GPUs $CUDA_VISIBLE_DEVICES"
echo "Auxiliary gradient checkpointing: $AUXILIARY_GRADIENT_CHECKPOINTING"
echo "Checkpoints: $OUTPUT_DIR; logs: $LOG_ROOT/$RUN_ID"

exec bash "$PROJECT_ROOT/scripts/train_llama1b_joint_simcot.sh" \
  "$@" \
  --fsdp \
  --bf16 \
  --no-fp16 \
  --tf32 \
  "$auxiliary_checkpoint_arg"

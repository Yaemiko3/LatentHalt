#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${LATENTHALT_STORAGE_ROOT:-$PROJECT_ROOT}"
WORKSPACE_ROOT="$(dirname -- "$PROJECT_ROOT")"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-$PROJECT_ROOT/src/train_llama1b_simcot.py}"

SFT_MODEL_PATH="${SFT_MODEL_PATH:-$STORAGE_ROOT/outputs/sft-cot-llama1b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$SFT_MODEL_PATH}"
AUXILIARY_MODEL_PATH="${AUXILIARY_MODEL_PATH:-$WORKSPACE_ROOT/models/Llama-3.2-1B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$WORKSPACE_ROOT/datasets/gsm8k-aug/data/train-00000-of-00001.parquet}"
VALIDATION_FILE="${VALIDATION_FILE:-$WORKSPACE_ROOT/datasets/gsm8k-aug/data/validation-00000-of-00001.parquet}"
THINK_REGION_FILE="${THINK_REGION_FILE:-$PROJECT_ROOT/results/analysis/think_hidden_geometry/think_region.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/outputs/latent-halt-llama1b}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}"
LOG_ROOT="${LOG_ROOT:-$STORAGE_ROOT/logs}"

CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#VISIBLE_GPU_IDS[@]}"

C_THOUGHT="${C_THOUGHT:-2}"
EPOCHS_PER_STAGE="${EPOCHS_PER_STAGE:-3}"
MAX_LATENT_STAGE="${MAX_LATENT_STAGE:-10}"
CURRICULUM_START_STAGE="${CURRICULUM_START_STAGE:-1}"
REGION_LOSS_WEIGHT="${REGION_LOSS_WEIGHT:-5.0}"
REGION_NEGATIVE_LOSS_WEIGHT="${REGION_NEGATIVE_LOSS_WEIGHT:-1.0}"
DECODER_LOSS_WEIGHT="${DECODER_LOSS_WEIGHT:-0.5}"
DECODER_LOSS_NORMALIZATION="${DECODER_LOSS_NORMALIZATION:-block}"
NUM_EPOCHS="${NUM_EPOCHS:-33}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-4}"
DECODER_LEARNING_RATE="${DECODER_LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
STAGE_LR_DECAY="${STAGE_LR_DECAY:-0}"
STAGE_LR_MIN_RATIO="${STAGE_LR_MIN_RATIO:-0.1}"
STAGE_LR_RAMP_RATIO="${STAGE_LR_RAMP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DECODER_MAX_LENGTH="${DECODER_MAX_LENGTH:-512}"
LENGTH_BUCKET_WIDTH="${LENGTH_BUCKET_WIDTH:-32}"
FIRST_LATENT_BUCKET_WIDTH="${FIRST_LATENT_BUCKET_WIDTH:-16}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
RESET_OPTIMIZER_EACH_EPOCH="${RESET_OPTIMIZER_EACH_EPOCH:-0}"
RUN_ID="${RUN_ID:-train-joint-simcot-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"

case "$RESET_OPTIMIZER_EACH_EPOCH" in
  1|true|TRUE|yes|YES)
    RESET_OPTIMIZER_ARG="--reset_optimizer_each_epoch"
    ;;
  0|false|FALSE|no|NO)
    RESET_OPTIMIZER_ARG="--no-reset_optimizer_each_epoch"
    ;;
  *)
    echo "RESET_OPTIMIZER_EACH_EPOCH must be a boolean; got: $RESET_OPTIMIZER_EACH_EPOCH" >&2
    exit 2
    ;;
esac

case "$STAGE_LR_DECAY" in
  1|true|TRUE|yes|YES)
    STAGE_LR_DECAY_ARG="--stage_lr_decay"
    ;;
  0|false|FALSE|no|NO)
    STAGE_LR_DECAY_ARG="--no-stage_lr_decay"
    ;;
  *)
    echo "STAGE_LR_DECAY must be a boolean; got: $STAGE_LR_DECAY" >&2
    exit 2
    ;;
esac

if [[ ! "$PER_DEVICE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "PER_DEVICE_BATCH_SIZE must be a positive integer; got: $PER_DEVICE_BATCH_SIZE" >&2
  exit 2
fi
resolve_resume_checkpoint_for_batch_alignment() {
  local requested_lower="${RESUME_FROM_CHECKPOINT,,}"
  local checkpoint_name

  case "$requested_lower" in
    ""|none|false)
      return 0
      ;;
    auto)
      [[ -d "$OUTPUT_DIR" ]] || return 0
      checkpoint_name="$({
        find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d \
          -name 'checkpoint-*' -printf '%f\n' 2>/dev/null || true
      } | sort -V | tail -n 1)"
      [[ -n "$checkpoint_name" ]] || return 0
      printf '%s\n' "$OUTPUT_DIR/$checkpoint_name"
      ;;
    *)
      realpath -m -- "$RESUME_FROM_CHECKPOINT"
      ;;
  esac
}

align_resume_checkpoint_batch_size() {
  local checkpoint state_file stored_batch_size backup_file temporary_file
  checkpoint="$(resolve_resume_checkpoint_for_batch_alignment)"
  [[ -n "$checkpoint" ]] || return 0

  state_file="$checkpoint/trainer_state.json"
  if [[ ! -f "$state_file" ]]; then
    echo "Resume checkpoint is missing trainer_state.json: $checkpoint" >&2
    exit 2
  fi
  if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to align the resume checkpoint batch size" >&2
    exit 2
  fi

  stored_batch_size="$(jq -r '.train_batch_size // empty' "$state_file")"
  if [[ "$stored_batch_size" == "$PER_DEVICE_BATCH_SIZE" ]]; then
    echo "Resume batch size is consistent: $PER_DEVICE_BATCH_SIZE ($checkpoint)"
    return 0
  fi

  backup_file="$state_file.before-batch-size-alignment"
  if [[ ! -e "$backup_file" ]]; then
    cp --preserve=mode,timestamps -- "$state_file" "$backup_file"
  fi
  temporary_file="$(mktemp "$checkpoint/.trainer_state.json.XXXXXX")"
  if ! jq --argjson batch_size "$PER_DEVICE_BATCH_SIZE" \
    '.train_batch_size = $batch_size' "$state_file" > "$temporary_file"; then
    rm -f -- "$temporary_file"
    echo "Failed to update resume checkpoint batch size: $state_file" >&2
    exit 2
  fi
  chmod --reference="$state_file" "$temporary_file"
  mv -f -- "$temporary_file" "$state_file"
  echo "Aligned resume batch size: ${stored_batch_size:-unset} -> $PER_DEVICE_BATCH_SIZE ($checkpoint)"
  echo "Original trainer state: $backup_file"
}

if [[ ! -d "$SFT_MODEL_PATH" ]]; then
  echo "Missing SFT model directory: $SFT_MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$TOKENIZER_PATH" ]]; then
  echo "Missing tokenizer directory: $TOKENIZER_PATH" >&2
  exit 2
fi
if [[ ! -d "$AUXILIARY_MODEL_PATH" ]]; then
  echo "Missing auxiliary decoder initialization model: $AUXILIARY_MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Missing joint SIM-CoT training parquet: $TRAIN_FILE" >&2
  exit 2
fi
if [[ ! -f "$VALIDATION_FILE" ]]; then
  echo "Missing joint SIM-CoT validation parquet: $VALIDATION_FILE" >&2
  exit 2
fi
if [[ ! -f "$THINK_REGION_FILE" ]]; then
  echo "Missing think-region artifact: $THINK_REGION_FILE" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_ENTRYPOINT" ]]; then
  echo "Missing SIM-CoT training entrypoint: $TRAIN_ENTRYPOINT" >&2
  exit 2
fi
align_resume_checkpoint_batch_size

mkdir -p "$RUN_LOG_DIR" "$LOG_ROOT"
export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LATENTHALT_LOG_ROOT="$LOG_ROOT"
export LATENTHALT_RUN_ID="$RUN_ID"
export LATENTHALT_RUN_STARTED_AT_UNIX="$(date +%s.%N)"
export RUN_PLATFORM_TYPE="${RUN_PLATFORM_TYPE:-local_server}"
export RUN_PLATFORM_NAME="${RUN_PLATFORM_NAME:-local}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=1
fi

effective_batch_size=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS processes)"
echo "CUDA device order: $CUDA_DEVICE_ORDER"
echo "Effective batch size: $PER_DEVICE_BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS * $NUM_GPUS = $effective_batch_size"
echo "SFT base model: $SFT_MODEL_PATH"
echo "Auxiliary decoder initialization: $AUXILIARY_MODEL_PATH"
echo "Loss: language model + $REGION_LOSS_WEIGHT * final-region linear hinge + $REGION_NEGATIVE_LOSS_WEIGHT * premature-region squared hinge + $DECODER_LOSS_WEIGHT * decoder ($DECODER_LOSS_NORMALIZATION normalization)"
echo "Learning rates: base peak=$BASE_LEARNING_RATE, auxiliary decoder peak=$DECODER_LEARNING_RATE"
echo "Stage LR schedule: enabled=$STAGE_LR_DECAY, min ratio=$STAGE_LR_MIN_RATIO, ramp ratio=$STAGE_LR_RAMP_RATIO; stage $MAX_LATENT_STAGE and fully latent share $((2 * EPOCHS_PER_STAGE)) epochs"
echo "Curriculum: prefix replacement, start stage $CURRICULUM_START_STAGE, +1 block every $EPOCHS_PER_STAGE epoch(s), cap $MAX_LATENT_STAGE blocks"

set +e
"${PYTHON:-python}" -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc_per_node "$NUM_GPUS" \
  "$TRAIN_ENTRYPOINT" \
    --init_from_sft \
    --sft_model_path "$SFT_MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --auxiliary_model_path "$AUXILIARY_MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VALIDATION_FILE" \
    --think_region_file "$THINK_REGION_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --log_dir "$LOG_ROOT" \
    --c_thought "$C_THOUGHT" \
    --epochs_per_stage "$EPOCHS_PER_STAGE" \
    --max_latent_stage "$MAX_LATENT_STAGE" \
    --curriculum_start_stage "$CURRICULUM_START_STAGE" \
    --region_loss_weight "$REGION_LOSS_WEIGHT" \
    --region_negative_loss_weight "$REGION_NEGATIVE_LOSS_WEIGHT" \
    --decoder_loss_weight "$DECODER_LOSS_WEIGHT" \
    --decoder_loss_normalization "$DECODER_LOSS_NORMALIZATION" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --base_learning_rate "$BASE_LEARNING_RATE" \
    --decoder_learning_rate "$DECODER_LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    "$STAGE_LR_DECAY_ARG" \
    --stage_lr_min_ratio "$STAGE_LR_MIN_RATIO" \
    --stage_lr_ramp_ratio "$STAGE_LR_RAMP_RATIO" \
    --max_length "$MAX_LENGTH" \
    --decoder_max_length "$DECODER_MAX_LENGTH" \
    --length_bucket_width "$LENGTH_BUCKET_WIDTH" \
    --first_latent_bucket_width "$FIRST_LATENT_BUCKET_WIDTH" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
    "$RESET_OPTIMIZER_ARG" \
    "$@" \
    2>&1 | tee "$RUN_LOG_DIR/console.log"
exit_code=${PIPESTATUS[0]}
set -e

printf '%s\n' "$exit_code" > "$RUN_LOG_DIR/exit_code"
exit "$exit_code"

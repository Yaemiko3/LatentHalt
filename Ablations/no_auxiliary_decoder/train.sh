#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(dirname -- "$PROJECT_ROOT")"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-$PROJECT_ROOT/src/train_llama1b_simcot.py}"

SFT_MODEL_PATH="${SFT_MODEL_PATH:-$PROJECT_ROOT/outputs/sft-cot-llama1b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$SFT_MODEL_PATH}"
TRAIN_FILE="${TRAIN_FILE:-$WORKSPACE_ROOT/datasets/gsm8k-aug/data/train-00000-of-00001.parquet}"
VALIDATION_FILE="${VALIDATION_FILE:-$WORKSPACE_ROOT/datasets/gsm8k-aug/data/validation-00000-of-00001.parquet}"
THINK_REGION_FILE="${THINK_REGION_FILE:-$PROJECT_ROOT/results/analysis/think_hidden_geometry/think_region.safetensors}"
# Reuse the repository Hugging Face cache so this ablation does not duplicate
# the multi-gigabyte parquet conversion already used by the other runs.
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}"
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
NUM_EPOCHS="${NUM_EPOCHS:-33}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1e-4}"
DECODER_LEARNING_RATE="${DECODER_LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DECODER_MAX_LENGTH="${DECODER_MAX_LENGTH:-512}"
LENGTH_BUCKET_WIDTH="${LENGTH_BUCKET_WIDTH:-32}"
FIRST_LATENT_BUCKET_WIDTH="${FIRST_LATENT_BUCKET_WIDTH:-16}"
SAVE_STEPS="${SAVE_STEPS:-500}"
RESET_OPTIMIZER_EACH_EPOCH="${RESET_OPTIMIZER_EACH_EPOCH:-0}"

case "$RESET_OPTIMIZER_EACH_EPOCH" in
  1|true|TRUE|yes|YES) RESET_OPTIMIZER_ARG="--reset_optimizer_each_epoch" ;;
  0|false|FALSE|no|NO) RESET_OPTIMIZER_ARG="--no-reset_optimizer_each_epoch" ;;
  *) echo "RESET_OPTIMIZER_EACH_EPOCH must be a boolean; got: $RESET_OPTIMIZER_EACH_EPOCH" >&2; exit 2 ;;
esac

# Group this ablation under the project output and log directories.
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/ablations/no_auxiliary_decoder}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/ablations/no_auxiliary_decoder}"
RUN_ID="${RUN_ID:-no-auxiliary-decoder-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# Match the formal joint recipe and remove only auxiliary-decoder supervision.
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
STAGE_LR_DECAY="${STAGE_LR_DECAY:-0}"

case "$STAGE_LR_DECAY" in
  1|true|TRUE|yes|YES) STAGE_LR_DECAY_ARG="--stage_lr_decay" ;;
  0|false|FALSE|no|NO) STAGE_LR_DECAY_ARG="--no-stage_lr_decay" ;;
  *) echo "STAGE_LR_DECAY must be a boolean; got: $STAGE_LR_DECAY" >&2; exit 2 ;;
esac

if [[ ! -d "$SFT_MODEL_PATH" ]]; then
  echo "Missing SFT model directory: $SFT_MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$TOKENIZER_PATH" ]]; then
  echo "Missing tokenizer directory: $TOKENIZER_PATH" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" || ! -f "$VALIDATION_FILE" ]]; then
  echo "Missing joint SIM-CoT training or validation parquet" >&2
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

if [[ ! "$PER_DEVICE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "PER_DEVICE_BATCH_SIZE must be a positive integer; got: $PER_DEVICE_BATCH_SIZE" >&2
  exit 2
fi

RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$RUN_LOG_DIR"
export CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES
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
echo "Effective batch size: $PER_DEVICE_BATCH_SIZE * $GRADIENT_ACCUMULATION_STEPS * $NUM_GPUS = $effective_batch_size"
echo "SFT base model: $SFT_MODEL_PATH"
echo "Auxiliary decoder: disabled (no decoder model will be loaded or saved)"
echo "Loss: language model + $REGION_LOSS_WEIGHT * final-region linear hinge + $REGION_NEGATIVE_LOSS_WEIGHT * premature-region squared hinge"

set +e
"${PYTHON:-python}" -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc_per_node "$NUM_GPUS" \
  "$TRAIN_ENTRYPOINT" \
    --init_from_sft \
    --sft_model_path "$SFT_MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
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
    --decoder_loss_weight 0 \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --base_learning_rate "$BASE_LEARNING_RATE" \
    --decoder_learning_rate "$DECODER_LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    "$STAGE_LR_DECAY_ARG" \
    --max_length "$MAX_LENGTH" \
    --decoder_max_length "$DECODER_MAX_LENGTH" \
    --length_bucket_width "$LENGTH_BUCKET_WIDTH" \
    --first_latent_bucket_width "$FIRST_LATENT_BUCKET_WIDTH" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
    "$RESET_OPTIMIZER_ARG" \
    "$@" \
    --no-use_auxiliary_decoder \
    --decoder_loss_weight 0 \
    --region_loss_weight "$REGION_LOSS_WEIGHT" \
    --region_negative_loss_weight "$REGION_NEGATIVE_LOSS_WEIGHT" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_ROOT" \
    --cache_dir "$CACHE_DIR" \
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
    2>&1 | tee "$RUN_LOG_DIR/console.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$RUN_LOG_DIR/exit_code"
exit "$status"

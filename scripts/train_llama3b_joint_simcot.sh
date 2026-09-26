#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${LATENTHALT_STORAGE_ROOT:-$PROJECT_ROOT}"
WORKSPACE_ROOT="$(dirname -- "$PROJECT_ROOT")"

MODEL_ROOT="${LLAMA3B_MODEL_ROOT:-$WORKSPACE_ROOT/models/Llama-3.2-3B-Instruct}"
SFT_MODEL_PATH="${LLAMA3B_SFT_MODEL_PATH:-$STORAGE_ROOT/outputs/sft-cot-llama3b}"
TOKENIZER_PATH="${LLAMA3B_TOKENIZER_PATH:-$SFT_MODEL_PATH}"
AUXILIARY_MODEL_PATH="${LLAMA3B_AUXILIARY_MODEL_PATH:-$MODEL_ROOT}"
TRAIN_FILE="${LLAMA3B_TRAIN_FILE:-$WORKSPACE_ROOT/datasets/gsm8k-aug/data/train-00000-of-00001.parquet}"
VALIDATION_FILE="${LLAMA3B_VALIDATION_FILE:-$WORKSPACE_ROOT/datasets/gsm8k-aug/data/validation-00000-of-00001.parquet}"
THINK_REGION_FILE="${LLAMA3B_THINK_REGION_FILE:-$PROJECT_ROOT/results/analysis/think_hidden_geometry_llama3b/think_region.safetensors}"

LR_TAG="${LR_TAG:-base${BASE_LEARNING_RATE:-5e-5}}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-5e-5}"
DECODER_LEARNING_RATE="${DECODER_LEARNING_RATE:-1e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/outputs/latent-halt-3b-${LR_TAG}-dec1e-5-b16-ga4-single-bucket}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}"
LOG_ROOT="${LOG_ROOT:-$STORAGE_ROOT/logs}"

CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
PYTHON="${PYTHON:-python}"

# Fixed single-card protocol: 16 x 4 = effective batch 64.
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
EPOCHS_PER_STAGE="${EPOCHS_PER_STAGE:-3}"
MAX_LATENT_STAGE="${MAX_LATENT_STAGE:-10}"
CURRICULUM_START_STAGE="${CURRICULUM_START_STAGE:-1}"
C_THOUGHT="${C_THOUGHT:-2}"
REGION_LOSS_WEIGHT="${REGION_LOSS_WEIGHT:-5.0}"
REGION_NEGATIVE_LOSS_WEIGHT="${REGION_NEGATIVE_LOSS_WEIGHT:-1.0}"
DECODER_LOSS_WEIGHT="${DECODER_LOSS_WEIGHT:-0.5}"
DECODER_LOSS_NORMALIZATION="${DECODER_LOSS_NORMALIZATION:-block}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
STAGE_LR_MIN_RATIO="${STAGE_LR_MIN_RATIO:-0.1}"
STAGE_LR_RAMP_RATIO="${STAGE_LR_RAMP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DECODER_MAX_LENGTH="${DECODER_MAX_LENGTH:-512}"
LENGTH_BUCKET_WIDTH="${LENGTH_BUCKET_WIDTH:-32}"
FIRST_LATENT_BUCKET_WIDTH="${FIRST_LATENT_BUCKET_WIDTH:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
# Keep at most this many checkpoints in the disk output directory.
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-16}"
SEED="${SEED:-11}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
RUN_ID="${RUN_ID:-train-joint-simcot-3b-${LR_TAG}-b16-ga4-single-bucket-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"

if [[ "${#VISIBLE_GPU_IDS[@]}" -ne 1 ]]; then
  echo "This 3B joint SIM-CoT entrypoint is single-card only." >&2
  echo "Use exactly one GPU in CUDA_VISIBLE_DEVICES, got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
if [[ ! "$PER_DEVICE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "PER_DEVICE_BATCH_SIZE must be a positive integer; got: $PER_DEVICE_BATCH_SIZE" >&2
  exit 2
fi
if [[ ! "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GRADIENT_ACCUMULATION_STEPS must be a positive integer; got: $GRADIENT_ACCUMULATION_STEPS" >&2
  exit 2
fi
if [[ ! "$SAVE_TOTAL_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAVE_TOTAL_LIMIT must be a positive integer; got: $SAVE_TOTAL_LIMIT" >&2
  exit 2
fi
for required_dir in "$SFT_MODEL_PATH" "$TOKENIZER_PATH" "$AUXILIARY_MODEL_PATH"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Required model directory is missing: $required_dir" >&2
    exit 2
  fi
done
for required_file in "$TRAIN_FILE" "$VALIDATION_FILE" "$THINK_REGION_FILE"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file is missing: $required_file" >&2
    exit 2
  fi
done
if [[ ! -f "$PROJECT_ROOT/src/train_llama1b_simcot.py" ]]; then
  echo "SIM-CoT training entrypoint is missing: $PROJECT_ROOT/src/train_llama1b_simcot.py" >&2
  exit 2
fi
if ! "$PYTHON" -c 'import datasets, pyarrow, torch, transformers' >/dev/null 2>&1; then
  echo "SIM-CoT dependencies are incomplete in PYTHON=$PYTHON." >&2
  exit 2
fi

mkdir -p "$RUN_LOG_DIR" "$LOG_ROOT" "$CACHE_DIR"
export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LATENTHALT_LOG_ROOT="$LOG_ROOT"
export LATENTHALT_RUN_ID="$RUN_ID"
export LATENTHALT_RUN_STARTED_AT_UNIX="$(date +%s.%N)"
export RUN_PLATFORM_TYPE="${RUN_PLATFORM_TYPE:-local_server}"
export RUN_PLATFORM_NAME="${RUN_PLATFORM_NAME:-local-3b}"

effective_batch_size=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
echo "SIM-CoT model: $SFT_MODEL_PATH"
echo "Auxiliary decoder: $AUXILIARY_MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES (single process)"
echo "Per-device batch: train=$PER_DEVICE_BATCH_SIZE eval=$PER_DEVICE_EVAL_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective batch: $effective_batch_size"
echo "Layout bucketing: enabled (length=$LENGTH_BUCKET_WIDTH, first-latent=$FIRST_LATENT_BUCKET_WIDTH)"
echo "Learning rates: base=$BASE_LEARNING_RATE, decoder=$DECODER_LEARNING_RATE"
echo "Curriculum: $EPOCHS_PER_STAGE epochs/stage, stages $CURRICULUM_START_STAGE-$MAX_LATENT_STAGE, total $NUM_EPOCHS epochs"

set +e
"$PYTHON" "$PROJECT_ROOT/src/train_llama1b_simcot.py" \
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
  --region_positive_loss_type linear \
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
  --no-stage_lr_decay \
  --stage_lr_min_ratio "$STAGE_LR_MIN_RATIO" \
  --stage_lr_ramp_ratio "$STAGE_LR_RAMP_RATIO" \
  --max_length "$MAX_LENGTH" \
  --decoder_max_length "$DECODER_MAX_LENGTH" \
  --bucket_by_layout \
  --length_bucket_width "$LENGTH_BUCKET_WIDTH" \
  --first_latent_bucket_width "$FIRST_LATENT_BUCKET_WIDTH" \
  --logging_steps "$LOGGING_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --eval_strategy epoch \
  --dataloader_num_workers "$NUM_WORKERS" \
  --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
  --seed "$SEED" \
  --report_to tensorboard \
  --attn_implementation sdpa \
  --bf16 \
  --tf32 \
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
  --no-reset_optimizer_each_epoch \
  "$@" \
  2>&1 | tee "$RUN_LOG_DIR/console.log"
exit_code=${PIPESTATUS[0]}
set -e

printf '%s\n' "$exit_code" > "$RUN_LOG_DIR/exit_code"
exit "$exit_code"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${LATENTHALT_STORAGE_ROOT:-$PROJECT_ROOT}"
WORKSPACE_ROOT="$(dirname -- "$PROJECT_ROOT")"
MODEL_ROOT="${MODEL_ROOT:-$WORKSPACE_ROOT/models/Llama-3.2-3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/outputs/sft-cot-llama3b}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}"
LOG_ROOT="${LOG_ROOT:-$STORAGE_ROOT/logs}"

CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPU_IDS[@]}}"
PYTHON="${PYTHON:-python}"

# 64 examples x 1 accumulation step x 1 GPU = global batch 64.
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-16}"
SEED="${SEED:-11}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
RUN_ID="${RUN_ID:-train-sft-cot-llama3b-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"

if [[ ! -d "$MODEL_PATH" || ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Llama-3.2-3B model directory is missing config.json: $MODEL_PATH" >&2
  exit 2
fi
for required_file in \
  "$DATA_ROOT/gsm8k-aug/data/train-00000-of-00001.parquet" \
  "$DATA_ROOT/gsm8k-aug/data/validation-00000-of-00001.parquet"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required GSM8K-Aug file is missing: $required_file" >&2
    exit 2
  fi
done
if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 2
fi
if [[ "${#VISIBLE_GPU_IDS[@]}" -ne "$NUM_GPUS" ]]; then
  echo "NUM_GPUS=$NUM_GPUS does not match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  echo "Set both variables consistently, for example NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1." >&2
  exit 2
fi

mkdir -p "$RUN_LOG_DIR" "$LOG_ROOT" "$CACHE_DIR"
export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LATENTHALT_LOG_ROOT="$LOG_ROOT"
export LATENTHALT_RUN_ID="$RUN_ID"
export LATENTHALT_RUN_STARTED_AT_UNIX="$(date +%s.%N)"
export RUN_PLATFORM_TYPE="${RUN_PLATFORM_TYPE:-local_server}"
export RUN_PLATFORM_NAME="${RUN_PLATFORM_NAME:-local-3b}"

effective_batch_size=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
echo "SFT model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS processes)"
echo "Per-device batch: train=$PER_DEVICE_BATCH_SIZE eval=$PER_DEVICE_EVAL_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective global batch: $effective_batch_size"
echo "Learning rate: $LEARNING_RATE; epochs: $NUM_EPOCHS; max length: $MAX_LENGTH"

set +e
"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc_per_node "$NUM_GPUS" \
  "$PROJECT_ROOT/src/train_sft_cot.py" \
  --model_path "$MODEL_PATH" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  --log_dir "$LOG_ROOT" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_train_epochs "$NUM_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --warmup_ratio "$WARMUP_RATIO" \
  --max_length "$MAX_LENGTH" \
  --logging_steps "$LOGGING_STEPS" \
  --dataloader_num_workers "$NUM_WORKERS" \
  --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
  --seed "$SEED" \
  --attn_implementation sdpa \
  --bf16 \
  --tf32 \
  --gradient_checkpointing \
  --eval_strategy epoch \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
  "$@" \
  2>&1 | tee "$RUN_LOG_DIR/console.log"
exit_code=${PIPESTATUS[0]}
set -e

printf '%s\n' "$exit_code" > "$RUN_LOG_DIR/exit_code"
(( exit_code == 0 )) || exit "$exit_code"

CUDA_VISIBLE_DEVICES="${EVAL_GPU:-${VISIBLE_GPU_IDS[0]}}" \
MODEL_PATH="$OUTPUT_DIR" \
RESULTS_DIR="${RESULTS_DIR:-$STORAGE_ROOT/results/$(basename -- "$OUTPUT_DIR")}" \
DATA_ROOT="$DATA_ROOT" CACHE_DIR="$CACHE_DIR" LOG_ROOT="$LOG_ROOT" PYTHON="$PYTHON" \
RUN_ID="eval-$RUN_ID" \
  bash "$PROJECT_ROOT/scripts/eval_llama1b_sft_cot.sh" \
    --datasets gsm8k gsm-hard multi-arith svamp

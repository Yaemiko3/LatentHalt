#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${LATENTHALT_STORAGE_ROOT:-$PROJECT_ROOT}"
WORKSPACE_ROOT="$(dirname -- "$PROJECT_ROOT")"

MODEL_PATH="${MODEL_PATH:-$WORKSPACE_ROOT/models/Llama-3.2-1B-Instruct}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/outputs/sft-cot-llama1b}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}"
LOG_ROOT="${LOG_ROOT:-$STORAGE_ROOT/logs}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
NUM_GPUS="${NUM_GPUS:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
RUN_ID="${RUN_ID:-train-sft-cot-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"

mkdir -p "$RUN_LOG_DIR" "$LOG_ROOT"
export CUDA_DEVICE_ORDER
export LATENTHALT_LOG_ROOT="$LOG_ROOT"
export LATENTHALT_RUN_ID="$RUN_ID"
export LATENTHALT_RUN_STARTED_AT_UNIX="$(date +%s.%N)"
export RUN_PLATFORM_TYPE="${RUN_PLATFORM_TYPE:-local_server}"
export RUN_PLATFORM_NAME="${RUN_PLATFORM_NAME:-local}"

set +e
"${PYTHON:-python}" -m torch.distributed.run \
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
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_train_epochs "$NUM_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --max_length "$MAX_LENGTH" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
  "$@" \
  2>&1 | tee "$RUN_LOG_DIR/console.log"
exit_code=${PIPESTATUS[0]}
set -e

printf '%s\n' "$exit_code" > "$RUN_LOG_DIR/exit_code"
exit "$exit_code"

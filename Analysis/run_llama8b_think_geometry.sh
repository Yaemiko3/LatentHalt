#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${LATENTHALT_STORAGE_ROOT:-$PROJECT_ROOT}"
MODEL_PATH="${LLAMA8B_SFT_MODEL_PATH:-${MODEL_PATH:-$STORAGE_ROOT/outputs/sft-cot-llama8b}}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/results/analysis/think_hidden_geometry_llama8b}"
LOG_ROOT="${LOG_ROOT:-$STORAGE_ROOT/logs/analysis}"
RUN_ID="${RUN_ID:-think-geometry-llama8b-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "8B SFT model is missing config.json: $MODEL_PATH" >&2
  echo "Run scripts/train_llama8b_sft_cot.sh first." >&2
  exit 2
fi

mkdir -p "$RUN_LOG_DIR"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set +e
MODEL_PATH="$MODEL_PATH" OUTPUT_DIR="$OUTPUT_DIR" BATCH_SIZE="${BATCH_SIZE:-8}" \
  bash "$PROJECT_ROOT/Analysis/run_llama1b_think_geometry.sh" \
    "$@" \
    2>&1 | tee "$RUN_LOG_DIR/console.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$RUN_LOG_DIR/exit_code"
exit "$status"

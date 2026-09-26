#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/ablations/squared_hinge/base_model}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/ablations/squared_hinge}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/ablations/squared_hinge}"
RUN_ID="${RUN_ID:-hinge-squared-eval-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

exec env \
  MODEL_PATH="$MODEL_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  RESULTS_DIR="$RESULTS_DIR" \
  LOG_DIR="$LOG_ROOT" \
  RUN_ID="$RUN_ID" \
  bash "$PROJECT_ROOT/scripts/eval_llama1b_latent.sh" \
  "$@" \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output_dir "$RESULTS_DIR" \
  --log_dir "$LOG_ROOT"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/ablations/no_geometry/base_model}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/ablations/no_geometry}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/ablations/no_geometry}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ID="${RUN_ID:-no-geometry-eval-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

# The q95 region is deliberately used only as an inference-time probe here;
# training never receives either side of the region loss.
exec env \
  MODEL_PATH="$MODEL_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  RESULTS_DIR="$RESULTS_DIR" \
  LOG_DIR="$LOG_DIR" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  RUN_ID="$RUN_ID" \
  bash "$PROJECT_ROOT/scripts/eval_llama1b_latent.sh" "$@"

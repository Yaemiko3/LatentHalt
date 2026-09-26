#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/ablations/no_auxiliary_decoder/base_model}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/ablations/no_auxiliary_decoder}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/ablations/no_auxiliary_decoder}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ID="${RUN_ID:-no-auxiliary-decoder-eval-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

# The inference code already evaluates only the exported base_model. Keeping
# this wrapper parallel to the other ablations makes the artifact layout and
# evaluation settings directly comparable.
exec env \
  MODEL_PATH="$MODEL_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  RESULTS_DIR="$RESULTS_DIR" \
  LOG_DIR="$LOG_DIR" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  RUN_ID="$RUN_ID" \
  bash "$PROJECT_ROOT/scripts/eval_llama1b_latent.sh" "$@"

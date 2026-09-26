#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname -- "$PROJECT_ROOT")"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/coconut-llama1b/base_model}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
THINK_REGION_FILE="${THINK_REGION_FILE:-$PROJECT_ROOT/results/analysis/think_hidden_geometry/think_region.safetensors}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/datasets}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/coconut-llama1b}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MIN_LATENT_BLOCKS="${MIN_LATENT_BLOCKS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ID="${RUN_ID:-eval-coconut-llama1b-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Latent model directory does not exist: $MODEL_PATH" >&2
    echo "Wait for Coconut export or set MODEL_PATH to a standard base_model directory." >&2
    exit 1
fi
if [[ ! -d "$TOKENIZER_PATH" ]]; then
    echo "Tokenizer directory does not exist: $TOKENIZER_PATH" >&2
    exit 1
fi
if [[ ! -f "$THINK_REGION_FILE" ]]; then
    echo "Think-region artifact does not exist: $THINK_REGION_FILE" >&2
    exit 1
fi

mkdir -p "$CACHE_DIR" "$RESULTS_DIR" "$RUN_LOG_DIR"
export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export LATENTHALT_LOG_ROOT="$LOG_DIR"
export LATENTHALT_RUN_ID="$RUN_ID"
export LATENTHALT_RUN_STARTED_AT_UNIX="$(date +%s.%N)"
export RUN_PLATFORM_TYPE="${RUN_PLATFORM_TYPE:-local_server}"
export RUN_PLATFORM_NAME="${RUN_PLATFORM_NAME:-local}"

optional_args=()
if [[ -n "${C_THOUGHT:-}" ]]; then
    optional_args+=(--c_thought "$C_THOUGHT")
fi
if [[ -n "${MAX_LATENT_BLOCKS:-}" ]]; then
    optional_args+=(--max_latent_blocks "$MAX_LATENT_BLOCKS")
fi
if [[ -n "${HALT_THRESHOLD:-}" ]]; then
    optional_args+=(--halt_threshold "$HALT_THRESHOLD")
fi

log_file="$RUN_LOG_DIR/console.log"
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES; CUDA device order: $CUDA_DEVICE_ORDER"
echo "Latent stopping: q95 region boundary, checked once per complete block"

set +e
"${PYTHON:-python}" "$PROJECT_ROOT/src/eval_latent.py" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --think_region_file "$THINK_REGION_FILE" \
    --data_root "$DATA_ROOT" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$RESULTS_DIR" \
    --log_dir "$LOG_DIR" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --min_latent_blocks "$MIN_LATENT_BLOCKS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    "${optional_args[@]}" \
    "$@" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$RUN_LOG_DIR/exit_code"

if [[ $status -ne 0 ]]; then
    echo "Latent evaluation failed with status $status. Log: $log_file" >&2
    exit "$status"
fi
echo "Latent evaluation completed. Log: $log_file"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname -- "$PROJECT_ROOT")"
ANALYSIS_ROOT="$PROJECT_ROOT/Analysis"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/analysis/think_hidden_geometry}"

overwrite_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

"${PYTHON:-python}" "$ANALYSIS_ROOT/extract_think_hidden.py" \
  --model_path "${MODEL_PATH:-$PROJECT_ROOT/outputs/sft-cot-llama1b}" \
  --data_root "${DATA_ROOT:-$WORKSPACE_ROOT/datasets}" \
  --cache_dir "${CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "${BATCH_SIZE:-32}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --folds "${CV_FOLDS:-5}" \
  --seed "${SEED:-20260730}" \
  --min_token_frequency "${MIN_TOKEN_FREQUENCY:-50}" \
  --recall_target "${RECALL_TARGET:-0.95}" \
  --matrix_chunk_size "${MATRIX_CHUNK_SIZE:-4096}" \
  --pca_device "${PCA_DEVICE:-auto}" \
  "${overwrite_args[@]}" \
  "$@"

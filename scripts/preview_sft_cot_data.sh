#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname -- "$PROJECT_ROOT")"

"${PYTHON:-python}" "$PROJECT_ROOT/src/train_sft_cot.py" \
  --model_path "${MODEL_PATH:-$WORKSPACE_ROOT/models/Llama-3.2-1B-Instruct}" \
  --data_root "${DATA_ROOT:-$WORKSPACE_ROOT/datasets}" \
  --cache_dir "${CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES:-8}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-8}" \
  --preprocessing_num_workers 1 \
  --preview_samples "${PREVIEW_SAMPLES:-2}" \
  --report_to none \
  --no-formal_experiment \
  --dry_run \
  "$@"

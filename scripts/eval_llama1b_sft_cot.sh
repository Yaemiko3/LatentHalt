#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname -- "$PROJECT_ROOT")"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
RUN_ID="${RUN_ID:-eval-sft-cot-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/outputs/sft-cot-llama1b}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/datasets}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/.cache/huggingface}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/sft-cot-llama1b}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

mkdir -p "$RUN_LOG_DIR"
export LATENTHALT_LOG_ROOT="$LOG_ROOT"
export LATENTHALT_RUN_ID="$RUN_ID"
export LATENTHALT_RUN_STARTED_AT_UNIX="$(date +%s.%N)"
export RUN_PLATFORM_TYPE="${RUN_PLATFORM_TYPE:-local_server}"
export RUN_PLATFORM_NAME="${RUN_PLATFORM_NAME:-local}"

set +e
"${PYTHON:-python}" "$PROJECT_ROOT/src/eval_math.py" \
  --model_path "$MODEL_PATH" \
  --data_root "$DATA_ROOT" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$RESULTS_DIR" \
  --log_dir "$LOG_ROOT" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  "$@" \
  2>&1 | tee "$RUN_LOG_DIR/console.log"
exit_code=${PIPESTATUS[0]}
set -e

printf '%s\n' "$exit_code" > "$RUN_LOG_DIR/exit_code"
if [[ "$exit_code" -eq 0 ]]; then
  flops_summary_path="$RUN_LOG_DIR/flops_summary.json"
  "${PYTHON:-python}" - "$RESULTS_DIR/summary.json" "$flops_summary_path" <<'PY' | tee -a "$RUN_LOG_DIR/console.log"
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
with summary_path.open("r", encoding="utf-8") as handle:
    summary = json.load(handle)
overall = summary.get("overall", {})
total_flops = overall.get("total_flops_excluding_prefill", 0)
average_flops = overall.get("average_flops_per_question_excluding_prefill", 0.0)
record = {
    "scope": (
        "explicit CoT and answer cached decode after question prefill; first token, "
        "question prefill, and padding after EOS excluded"
    ),
    "questions": overall.get("total", 0),
    "total_flops_excluding_prefill": total_flops,
    "average_flops_per_question_excluding_prefill": average_flops,
}
output_path.write_text(
    json.dumps(record, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
)
print(
    "Average continuation FLOPs per question (question prefill excluded): "
    f"{average_flops:.0f}"
)
print(f"FLOPs summary: {output_path}")
PY
fi
exit "$exit_code"

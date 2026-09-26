#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# Keep this ablation self-contained and aligned with the formal joint recipe.
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/ablations/squared_hinge}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/ablations/squared_hinge}"
RUN_ID="${RUN_ID:-hinge-squared-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
STAGE_LR_DECAY="${STAGE_LR_DECAY:-0}"
REGION_LOSS_WEIGHT="${REGION_LOSS_WEIGHT:-5.0}"
REGION_NEGATIVE_LOSS_WEIGHT="${REGION_NEGATIVE_LOSS_WEIGHT:-1.0}"
DECODER_LOSS_WEIGHT="${DECODER_LOSS_WEIGHT:-0.5}"

# Put fixed ablation arguments after "$@". The selected objective and output
# location therefore cannot be changed accidentally by an extra CLI option.
exec env \
  OUTPUT_DIR="$OUTPUT_DIR" \
  LOG_ROOT="$LOG_ROOT" \
  RUN_ID="$RUN_ID" \
  RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" \
  SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
  PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
  GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
  STAGE_LR_DECAY="$STAGE_LR_DECAY" \
  REGION_LOSS_WEIGHT="$REGION_LOSS_WEIGHT" \
  REGION_NEGATIVE_LOSS_WEIGHT="$REGION_NEGATIVE_LOSS_WEIGHT" \
  DECODER_LOSS_WEIGHT="$DECODER_LOSS_WEIGHT" \
  bash "$PROJECT_ROOT/scripts/train_llama1b_joint_simcot.sh" \
  "$@" \
  --region_positive_loss_type squared \
  --region_loss_weight "$REGION_LOSS_WEIGHT" \
  --region_negative_loss_weight "$REGION_NEGATIVE_LOSS_WEIGHT" \
  --decoder_loss_weight "$DECODER_LOSS_WEIGHT" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_ROOT" \
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"

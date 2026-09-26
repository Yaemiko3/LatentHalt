#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# Group this ablation under the project output and log directories.
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/ablations/no_geometry}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/ablations/no_geometry}"
RUN_ID="${RUN_ID:-no-geometry-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# Match the formal 1B run rather than the generic launcher's evolving defaults.
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
STAGE_LR_DECAY="${STAGE_LR_DECAY:-0}"

# The two region terms are forced off after user arguments so that this
# experiment cannot accidentally retain either terminal or nonterminal geometry
# supervision. All other options are inherited from the main joint recipe.
exec env \
  REGION_LOSS_WEIGHT=0 \
  REGION_NEGATIVE_LOSS_WEIGHT=0 \
  OUTPUT_DIR="$OUTPUT_DIR" \
  LOG_ROOT="$LOG_ROOT" \
  RUN_ID="$RUN_ID" \
  RESUME_FROM_CHECKPOINT="$RESUME_FROM_CHECKPOINT" \
  SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
  PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
  GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
  STAGE_LR_DECAY="$STAGE_LR_DECAY" \
  bash "$PROJECT_ROOT/scripts/train_llama1b_joint_simcot.sh" \
  "$@" \
  --region_loss_weight 0 \
  --region_negative_loss_weight 0

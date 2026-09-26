#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env BASE_LEARNING_RATE=5e-5 LR_TAG=base5e-5 \
  "$SCRIPT_DIR/train_llama3b_joint_simcot.sh" "$@"

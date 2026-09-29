#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Source cedia/config.local.sh first}"
: "${MANIFEST:?Source cedia/config.local.sh first}"
: "${RUNS:?Source cedia/config.local.sh first}"
FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:-1}}"
printf -v CHECKPOINT 'MODELS/best_metric_model_fold_%02d.pth' "$FOLD"
python -m cedia.cli evaluate --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --fold "$FOLD" --arch unet --name unet_original --checkpoint "$CHECKPOINT" \
  --output "$RUNS/eval/unet_original" --save-predictions --resume

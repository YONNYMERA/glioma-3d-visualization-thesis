#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${MANIFEST:?Set MANIFEST}"
: "${RUNS:?Set RUNS}"
: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"
FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:-1}}"
BASE="${DATASET_BASE:-700}"
TRAINER="${TRAINER:-nnUNetTrainer_100epochs}"
printf -v DATASET 'Dataset%03d_GliomaOuter%d' "$((BASE+FOLD))" "$FOLD"
python -m cedia.cli nnunet-train --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --fold "$FOLD" --dataset-dir "$nnUNet_raw/$DATASET" --trainer "$TRAINER" --resume
python -m cedia.cli nnunet-predict --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --fold "$FOLD" --dataset-dir "$nnUNet_raw/$DATASET" \
  --trained-model "$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__3d_fullres" \
  --output "$RUNS/nnunet_predictions" --resume
python -m cedia.cli nnunet-evaluate --data-root "$DATA_ROOT" --manifest "$MANIFEST" \
  --fold "$FOLD" --prediction-root "$RUNS/nnunet_predictions" \
  --output "$RUNS/eval/nnunet" --resume

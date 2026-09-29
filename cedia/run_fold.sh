#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${MANIFEST:?Set MANIFEST}"
: "${RUNS:?Set RUNS}"
ARCH="${ARCH:-unet}"
PROFILE="${PROFILE:-baseline}"
FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:-1}}"
WORKERS="${WORKERS:-4}"
case "$PROFILE" in
    baseline) OPTIONS=(--epochs 20 --scheduler none --patience 0) ;;
    extended) OPTIONS=(--epochs 100 --scheduler plateau --patience 15) ;;
    *) echo "PROFILE must be baseline or extended" >&2; exit 2 ;;
esac
NAME="${ARCH}_${PROFILE}"
python -m cedia.cli train --data-root "$DATA_ROOT" --manifest "$MANIFEST" --fold "$FOLD" \
    --arch "$ARCH" --output "$RUNS/train/$NAME" --workers "$WORKERS" "${OPTIONS[@]}" --resume
python -m cedia.cli evaluate --data-root "$DATA_ROOT" --manifest "$MANIFEST" --fold "$FOLD" \
    --arch "$ARCH" --name "$NAME" --checkpoint "$RUNS/train/$NAME/fold_$(printf '%02d' "$FOLD")/best.pth" \
    --output "$RUNS/eval/$NAME" --save-predictions --resume

#!/usr/bin/env bash
# Copy to config.local.sh and edit DATA_ROOT_ORIGINAL before sourcing.
export PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_ROOT_ORIGINAL="/RUTA/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
export RUNS="$PROJECT_DIR/results_reproduction"
export DATA_ROOT="$RUNS/cohorte_original"
export MANIFEST="$PROJECT_DIR/provenance/manifest_por_paciente.json"
export VENV_DIR="$PROJECT_DIR/.venv"
export nnUNet_raw="$RUNS/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$RUNS/nnunet/nnUNet_preprocessed"
export nnUNet_results="$RUNS/nnunet/nnUNet_results"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export WORKERS=4
# Fill in your allocation/account/time options from the actual CEDIA service.
SLURM_OPTIONS=()

#!/usr/bin/env bash
# Supply actual cluster resources on the sbatch command line. No account/partition/module guessed.
# Example structure: sbatch [YOUR_CEDIA_RESOURCE_FLAGS] cedia/slurm_job.sh python -m cedia.cli ...
set -euo pipefail
: "${PROJECT_DIR:?Set PROJECT_DIR to the absolute project directory}"
: "${VENV_DIR:?Set VENV_DIR to the Python environment directory}"
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
if [[ $# -eq 0 ]]; then
    echo "Pass a command after cedia/slurm_job.sh" >&2
    exit 2
fi
exec "$@"

#!/bin/bash
#SBATCH --job-name=electra_MM
#SBATCH --mail-type=END,FAIL        # mail goes to the submitting user
#SBATCH --partition=sm3090el9_devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G

#SBATCH --output=logs_validation/electra_MM_%j.log

# Usage: sbatch validate_electra.sh [--config <yaml>] [--metrics_file <path>]
# Arguments are passed on to validate_mm.py (default config: hpc_conf_mol_mm_extra.yaml).

# Debugging flags (optional)

# Run from the directory the job was submitted from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

# KeOps needs CUDA; el8 nodes have 12.1.1, el9 nodes only 12.9.1.
module load CUDA/12.1.1 2>/dev/null || module load CUDA/12.9.1

# Conda environment (name, not path). conda must be on PATH when you submit.
CONDA_ENV=${CONDA_ENV:-electra_env}
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python3 validate_mm.py "$@"

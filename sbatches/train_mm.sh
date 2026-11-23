#!/bin/bash
#SBATCH --job-name=electra_MM
#SBATCH --mail-type=END,FAIL        # mail goes to the submitting user
#SBATCH --partition=sm3090el9,sm3090el9_768
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=07-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G

#SBATCH --output=logs/electra_MM_%j.log

# Multitask (energy + density) training with train2.py.
#
# Submit from the repo root:   sbatch sbatches/train_mm.sh
# All paths are relative to the repo. The only data outside it is the QM9
# density set (qm9_dens_path in hpc_conf_mol_mm.yaml, or ELECTRA_QM9_PATH).
# Weights are saved to trained_models/ in the repo after every epoch.

# ---- Per-run settings ------------------------------------------------------
# Slurm stores a copy of this script when you submit it, so changing these
# values does not affect jobs that are already queued. The YAML is only read
# when a job starts. Leave a value empty to use the one from the YAML.
SCRIPT=train2.py              # train.py (energy only) | train2.py (energy + density)

SEED=202
lr_id='normal'                # 'normal' (4e-6) | 'fine' (1e-7); sets lr_normal/lr_fine in the run name
INITIAL_LR=4e-6
FINAL_LR=4e-6
DATA_SPLIT=mpfull2025         # mpfull2025 | mp50%2025 | mp10%2025 | mp1%2025
LOAD_MODEL_PATH=              # checkpoint to resume from (relative to the repo root); empty = train from scratch
USE_FRACTION_OF_DENSITY=false # false = 100% density; true = only INDICES_FOR_DENSITY get density loss
DENSITY_FRACTION=
INDICES_FOR_DENSITY=          # indices_10pct.json | indices_1pct.json (in data_splits/density_indices/)
PREV_EPOCHS=                  # epochs finished by the run being resumed (adds prev<N>ep to the run name)
# ----------------------------------------------------------------------------

# Run from the directory the job was submitted from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
if [ ! -f "$SCRIPT" ]; then
    echo "$SCRIPT not found in $(pwd); submit this script from the repo root." >&2
    exit 1
fi

ARGS=()
[ -n "$SEED" ]                && ARGS+=(--seed "$SEED")
[ -n "$INITIAL_LR" ]          && ARGS+=(--initial_lr "$INITIAL_LR")
[ -n "$FINAL_LR" ]            && ARGS+=(--final_lr "$FINAL_LR")
[ -n "$lr_id" ]               && ARGS+=(--lr_id "$lr_id")
[ -n "$DATA_SPLIT" ]          && ARGS+=(--data_split "$DATA_SPLIT")
if [ -n "$LOAD_MODEL_PATH" ]; then
    ARGS+=(--load_model --load_model_path "$LOAD_MODEL_PATH")
else
    ARGS+=(--no_load_model)
fi
[ "$USE_FRACTION_OF_DENSITY" = true ]  && ARGS+=(--use_fraction_of_density)
[ "$USE_FRACTION_OF_DENSITY" = false ] && ARGS+=(--no_use_fraction_of_density)
[ -n "$DENSITY_FRACTION" ]    && ARGS+=(--density_fraction "$DENSITY_FRACTION")
[ -n "$INDICES_FOR_DENSITY" ] && ARGS+=(--indices_for_density "$INDICES_FOR_DENSITY")
[ -n "$PREV_EPOCHS" ]         && ARGS+=(--prev_epochs "$PREV_EPOCHS")

echo "Job $SLURM_JOB_ID on partition $SLURM_JOB_PARTITION, time limit $(squeue -h -j "$SLURM_JOB_ID" -o %l)"

# KeOps needs the CUDA libraries; without them it falls back to CPU and segfaults.
# el8 nodes have CUDA/12.1.1, el9 nodes only CUDA/12.9.1.
module load CUDA/12.1.1 2>/dev/null || module load CUDA/12.9.1
echo "CUDA_HOME=$CUDA_HOME"

# Conda environment (name, not path). conda must be on PATH when you submit.
CONDA_ENV=${CONDA_ENV:-electra_env}
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH" >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Don't let wandb copy stdout into output.log/.wandb; the Slurm log still has it.
export WANDB_CONSOLE=off
# Log to a shared wandb team instead of your personal account:
# export WANDB_ENTITY=<team-name>

# Optional: put validation outputs (CHGCARs, deltas, HTML plots) on a scratch
# disk instead of the repo. Heavy writes on a shared home filesystem can get
# jobs killed.
# export ELECTRA_OUTPUT_BASE=/path/to/scratch/ELECTRA/

# Keep wandb's local run files on the node's own disk. Metrics still sync to
# wandb.ai; the local copy is moved to WANDB_RUNS_DIR when the job ends.
WANDB_RUNS_DIR=${WANDB_RUNS_DIR:-$(pwd)/wandb_runs}
export WANDB_DIR=${TMPDIR:-/tmp}/wandb_${SLURM_JOB_ID:-$$}
mkdir -p "$WANDB_DIR"
trap 'mkdir -p "$WANDB_RUNS_DIR" && cp -r "$WANDB_DIR"/wandb/* "$WANDB_RUNS_DIR"/ 2>/dev/null; rm -rf "$WANDB_DIR"' EXIT

echo "Running $SCRIPT ${ARGS[*]}"
python3 "$SCRIPT" "${ARGS[@]}"

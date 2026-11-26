#!/bin/bash
#SBATCH --job-name=electra_MM
#SBATCH --mail-type=END,FAIL        # mail goes to the submitting user
#SBATCH --partition=sm3090el9
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G

#SBATCH --output=logs7/electra_MM_%j.log

# Energy-only ELECTRA (train.py + ELECTRA_energy.py), seed run for the MLST revision.
# Identical to the old seed-505 sibling (first run of clear-butterfly-225) except the seed.
# Queue with: LONG_PARTITIONS=sm3090el9 SHORT_PARTITIONS=a100el9,h200el9 ./submit.sh <this script>

# ---- Per-run settings ------------------------------------------------------
# Slurm stores a copy of this script when you submit it, so later edits to this
# script, to the YAML files or to the embedded config below do not affect jobs
# that are already queued. hpc_conf_mol_energy.yaml is NOT read by these jobs.
SCRIPT=train.py               # train.py (energy only) | train2.py (energy + density)

SEED=101
CHANNELS=100                 # hotpp_outdim = master_units
lr_id='normal'
INITIAL_LR=4e-6
FINAL_LR=4e-6
DATA_SPLIT=mp50%2025
LOAD_MODEL_PATH=""            # empty = from scratch
PREV_EPOCHS=                  # epochs finished by earlier runs (adds prev<N>ep to the run name)
# ----------------------------------------------------------------------------

# ---- All other settings (frozen at submission) -----------------------------
# Copy of hpc_conf_mol_energy.yaml as of 2026-11-25, minus the per-run keys set
# above: seed, data_split, initial_lr, final_lr, energy_initial_lr (= INITIAL_LR),
# hotpp_outdim and master_units (= CHANNELS), load_model (from LOAD_MODEL_PATH).
# train.py --config reads only this file.
CONFIG_FILE=${TMPDIR:-/tmp}/electra_config_${SLURM_JOB_ID:-$$}.yaml
cat > "$CONFIG_FILE" <<'END_OF_CONFIG'
eq_check: False
wandb: True

electra_type: single
cbf: False
# New stuff
activation_type: silu
plot_gaus_pos: True

max_time: '29:00:00:00'
#max_time: '00:00:02:00'
max_epochs: 1000

project_name: electra3
save_memory: True

MD: False
crystal: False
nmc: False

poly_order: 5

energy_channel_width: 100
use_extra_mlp: False

MD_Data: False
pbc: False
wrap: False
r_max: 3
weight_decay: 0
energy_weight_decay: 0.0
split_params: False
energy_coef: 10
hotpp_cutoff: 8
use_l1_energy_features: False
l1_energy_out_dim: 40
#xyz_structures_dir: "xyzfiles/xyzfiles"
n_warmups: 0
clip_grad: False
gradient_clip_value: 0.1
negative_contributions: True
use_pos_disp_functions: True
use_matrix_scalars: True
gaus_per_electrons: 90
lr_dec_every: 1
lr_gamma: 0.8
qm9_dens_path: /home/scratch3/jels/QM9_Full
inference_cutoff: 10.0
hotpp_nlayers: 6
hotpp_csv_dir: csvs2
prune: True

model_type: hotpp
miaonet: miaomiao
max_out_heads: [2,2,2] # scal, vec, tens
head_activate_list: ['silu', 'silu', 'silu']
#hotpp params
norm_blocks: 'graph'
conv_mode: node_j
update_edge: False
#norm_heads: ['graph', 'graph', 'graph']
norm_heads: ['graph','graph','graph']
n_max: 3
out_max: 3
embed_type: atomic
precision: float

# MODEL PARAMS
loss_type: "normal"
loss_coef: 10
huber_loss_threshold: 0.1
reduce_weight_by: 4
relu: True
energy_csv: data_splits/energies.csv

n_train_points: 500000
n_split_points: 500000
n_split_points_single: 500000
train_sample_all: True
train_set_perc: 0.9
n_files: 1500
max_grid_points: 10000

niflheim: True
hpc: True

# FOLDERS

data_split_path: data_splits/

hpc_base_path: ./
niflheim_base_path: ./
pred_dens_path_val: pred_dens/validation/
density_delta_path_val: density_deltas/validation
pred_dens_path_test: pred_dens/test/
density_delta_path_test: density_deltas/test
gaus_pos_path_val: gaussian_positions/validation/
gaus_pos_path_test: gaussian_positions/test/

model_dir: trained_models

# TRAINING
optimizer: adamw
n_epochs: 1000
real_batch_size: 1
num_batches_before_update: 1
eval_every: 1
run_validation: False   # 2026-11-25: no per-epoch validation (old energy runs never ran it); validate checkpoints with validate_energy.py
construct_test_cd: False
construct_val_cd: True
save_model: True
END_OF_CONFIG
# ----------------------------------------------------------------------------

ARGS=(--config "$CONFIG_FILE")
[ -n "$SEED" ]                && ARGS+=(--seed "$SEED")
[ -n "$CHANNELS" ]            && ARGS+=(--channels "$CHANNELS")
[ -n "$INITIAL_LR" ]          && ARGS+=(--initial_lr "$INITIAL_LR")
[ -n "$FINAL_LR" ]            && ARGS+=(--final_lr "$FINAL_LR")
[ -n "$lr_id" ]               && ARGS+=(--lr_id "$lr_id")
[ -n "$DATA_SPLIT" ]          && ARGS+=(--data_split "$DATA_SPLIT")
if [ -n "$LOAD_MODEL_PATH" ]; then
    ARGS+=(--load_model --load_model_path "$LOAD_MODEL_PATH")
else
    ARGS+=(--no_load_model)
fi
[ -n "$PREV_EPOCHS" ]         && ARGS+=(--prev_epochs "$PREV_EPOCHS")

# ---- Sibling copies (only for jobs queued with submit.sh) -------------------
# submit.sh queues a 7-day and a 2-day copy under the same PAIR_TAG. The first
# one to start cancels the other; if both start at once the lower job id wins.
if [ -n "${PAIR_TAG:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    while read -r other state; do
        [ "$other" = "$SLURM_JOB_ID" ] && continue
        if [ "$state" = RUNNING ] && [ "$other" -lt "$SLURM_JOB_ID" ]; then
            echo "Sibling job $other is already running; exiting."
            exit 0
        fi
        echo "Cancelling sibling job $other ($state)"
        scancel "$other"
    done < <(squeue -h -u "$USER" -n "$PAIR_TAG" -o "%i %T")
fi
echo "Job $SLURM_JOB_ID on partition $SLURM_JOB_PARTITION, time limit $(squeue -h -j "$SLURM_JOB_ID" -o %l)"

# Run from the directory the job was submitted from (the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

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

# Validation/test outputs go to scratch3, not /home/energy.
export ELECTRA_OUTPUT_BASE=${ELECTRA_OUTPUT_BASE:-/home/scratch3/$USER/ELECTRA/}

# Keep wandb's local run files on the node's own disk, not /home/energy. Heavy
# I/O on the group fileserver gets jobs killed. Metrics still sync to wandb.ai;
# the local copy is moved to $ELECTRA_OUTPUT_BASE/wandb_runs when the job ends.
export WANDB_DIR=${TMPDIR:-/tmp}/wandb_${SLURM_JOB_ID:-$$}
mkdir -p "$WANDB_DIR"
trap 'mkdir -p ${ELECTRA_OUTPUT_BASE}wandb_runs && cp -r "$WANDB_DIR"/wandb/* ${ELECTRA_OUTPUT_BASE}wandb_runs/ 2>/dev/null; rm -rf "$WANDB_DIR"; rm -f "$CONFIG_FILE"' EXIT

echo "Running $SCRIPT ${ARGS[*]}"
python3 "$SCRIPT" "${ARGS[@]}"

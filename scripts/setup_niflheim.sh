#!/bin/bash
# One-time setup of a fresh clone on Niflheim. Run from anywhere:
#   bash scripts/setup_niflheim.sh
#
# Checkpoints are ~3 GB and are rewritten every epoch. Writing them to the
# group home fileserver (/home/energy) has gotten jobs killed for heavy I/O, so
# trained_models/ becomes a symlink to scratch3. The job scripts also send
# validation outputs and wandb run files to the same scratch folder
# (ELECTRA_OUTPUT_BASE, default /home/scratch3/$USER/ELECTRA/).
set -euo pipefail
cd "$(dirname "$0")/.."

SCRATCH=${ELECTRA_OUTPUT_BASE:-/home/scratch3/$USER/ELECTRA/}
SCRATCH=${SCRATCH%/}

mkdir -p "$SCRATCH"/trained_models "$SCRATCH"/trained_interesting_models "$SCRATCH"/wandb_runs
for d in trained_models trained_interesting_models; do
    if [ -e "$d" ] && [ ! -L "$d" ]; then
        echo "$d/ already exists as a real folder; move its files to $SCRATCH/$d and rerun." >&2
        exit 1
    fi
    ln -sfn "$SCRATCH/$d" "$d"
    echo "$d -> $SCRATCH/$d"
done

# Slurm does not create the --output folder; a job with a missing one fails at once.
mkdir -p logs logs7 logs_validation

if ! command -v conda >/dev/null 2>&1; then
    echo "Note: conda is not on PATH. The job scripts need it (they run 'conda activate \$CONDA_ENV')." >&2
fi
echo "Done."

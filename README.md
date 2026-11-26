# Multi-task ELECTRA

**Multi-task ELECTRA** is a research project exploring equivariant graph neural networks for predicting electron density grids and molecular energies from atomic structures simultaneously. It is an extension of ELECTRA,
which is a state of the art model for predicting electron densities only.
The framework integrates a PyTorch Lightning–based ELECTRA model that constructs neighbor graphs from atomic configurations and learns density reconstruction and energy regression.

## Highlights
- Equivariant density modeling with scalar, vector, and tensor representations  
- Config-driven training for molecular systems 
- Designed for large-scale datasets and HPC workflows  

## Project Status
This repository reflects **active, unpublished research**.  
Details related to datasets, training procedures, and full model configurations are intentionally limited.

## Technologies
Python · PyTorch Lightning · Equivariant GNNs · ASE · Scientific Machine Learning

## License and Use
This project is intended for research and evaluation purposes.  
Please contact the author for collaboration or citation guidance.

## Running experiments

All paths are relative to the repo root; submit jobs from there. Data that is
too big for the repo stays outside it, with Niflheim defaults:

| What | Default | Override |
|---|---|---|
| QM9 densities | `/home/scratch3/jels/QM9_Full` (`qm9_dens_path` in the YAML) | `ELECTRA_QM9_PATH` |
| Validation outputs, wandb run files | `/home/scratch3/$USER/ELECTRA/` (set in the job scripts) | `ELECTRA_OUTPUT_BASE` |
| Conda environment | `electra_env` (by name; `conda` must be on `PATH`) | `CONDA_ENV` (a name or a path) |

Checkpoints (~3 GB) are rewritten in `trained_models/` every epoch. On
`/home/energy` the training job scripts refuse to start unless `trained_models`
is a symlink to scratch (made by `scripts/setup_niflheim.sh`); heavy writes on
that fileserver have gotten jobs killed. Resume/finetune job scripts load
specific checkpoints by absolute path (`LOAD_MODEL_PATH`).

### Setup (once per clone)

```bash
conda env create -f environment.yml     # exported from the env the runs use
bash scripts/setup_niflheim.sh          # trained_models/ -> scratch3, log folders
```

`environment.yml` is a `conda env export`; `pytorch3d` and the `torch-*` PyG
extensions may need wheels built for torch 2.2.1 and your CUDA version.

### Training

- `train2.py`: multitask (energy + density), base config `hpc_conf_mol_mm.yaml`.
  Template: `sbatches/train_mm.sh`. Run scripts: `mpfull_jobs/`, `mp50_jobs/`,
  `mp10_jobs/`, `mp1_jobs/`.
- `train.py`: energy only. Scripts in `energy_jobs/` embed their full config and
  pass it with `--config`, so no YAML is read at job start.

Each job script has a per-run block at the top (seed, LR, split, checkpoint to
load, `PREV_EPOCHS`). Submit a single job with `sbatch <script>`, or queue a
7-day and a 2-day copy (the first to start cancels the other) with:

```bash
LONG_PARTITIONS=sm3090el9 SHORT_PARTITIONS=a100el9,h200el9 ./submit.sh energy_jobs/eo_mp1_64ch_seed101.sh
```

Logs go to `logs7/electra_MM_<jobid>.log`.

### Validation

```bash
sbatch validate_electra.sh --config validation_configs/<run>_mpfull_val.yaml \
    --metrics_file logs_validation/metrics_<run>_mpfull_val.txt
```

`validate_mm.py` computes energy MAE/RMSE on the 10k-molecule mpfull validation
set for the checkpoint in `load_model_path`. Use `validate_energy.py` for
energy-only models.

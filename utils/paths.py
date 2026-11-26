"""Repo-relative path handling for the YAML configs.

Relative paths in a config are taken relative to the root of this repository,
so the code runs from any clone without editing absolute paths. Data that is
too big for the repo stays outside it:

- QM9 densities: ``qm9_dens_path`` in the config (the shared Niflheim copy by
  default). Set ``ELECTRA_QM9_PATH`` to use another copy.
- Validation/test outputs (CHGCARs, deltas, HTML plots): ``niflheim_base_path``,
  overridden by ``ELECTRA_OUTPUT_BASE`` (see the job scripts).
"""
import os

# Root of this repository (the parent of utils/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config keys that hold file or directory paths. Keys used with string
# concatenation (path + "name") keep their trailing slash.
PATH_KEYS = ("data_split_path", "energy_csv", "hotpp_csv_dir", "hpc_base_path",
             "niflheim_base_path", "model_dir", "bad_files_path",
             "rad_cosine_log_path", "load_model_path", "xyz_structures_dir")

QM9_PATH_ENV = "ELECTRA_QM9_PATH"
OUTPUT_BASE_ENV = "ELECTRA_OUTPUT_BASE"


def resolve_config_paths(config: dict, base: str = REPO_ROOT) -> dict:
    """Turn relative paths in ``config`` into absolute ones under ``base``."""
    if os.environ.get(QM9_PATH_ENV):
        config["qm9_dens_path"] = os.environ[QM9_PATH_ENV]
    for key in PATH_KEYS:
        value = config.get(key)
        if not isinstance(value, str) or not value or os.path.isabs(value):
            continue
        resolved = os.path.normpath(os.path.join(base, value))
        if value.endswith("/"):
            resolved += "/"
        config[key] = resolved
    return config


def apply_output_base(config: dict) -> dict:
    """Send validation/test outputs to $ELECTRA_OUTPUT_BASE if it is set.

    Heavy writes on a shared home filesystem can get jobs killed, so the job
    scripts point this at a scratch disk.
    """
    if os.environ.get(OUTPUT_BASE_ENV):
        config["niflheim_base_path"] = os.environ[OUTPUT_BASE_ENV].rstrip("/") + "/"
    return config

#!/bin/bash
# Shared environment activation for local cluster jobs.
#
# Default: Matthias' bwUniCluster conda environment.
# Override without editing scripts:
#   export ROOMCONN_CONDA_ENV=other_env_name_or_/path/to/env

set -e

ROOMCONN_CONDA_ENV="${ROOMCONN_CONDA_ENV:-roomconn}"
ROOMCONN_CONDA_ENV_PATH="${ROOMCONN_CONDA_ENV_PATH:-$HOME/.conda/envs/$ROOMCONN_CONDA_ENV}"

# In SLURM batch jobs on bwUniCluster, the `conda` command can exist but fail with
# "ModuleNotFoundError: No module named 'conda'" on some nodes. Directly sourcing
# the environment's activate script avoids calling conda at all and is more robust.
if [ -f "$ROOMCONN_CONDA_ENV_PATH/bin/activate" ]; then
  source "$ROOMCONN_CONDA_ENV_PATH/bin/activate"
elif [ -d "$ROOMCONN_CONDA_ENV" ] && [ -f "$ROOMCONN_CONDA_ENV/bin/activate" ]; then
  source "$ROOMCONN_CONDA_ENV/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ROOMCONN_CONDA_ENV"
else
  echo "ERROR: Could not activate conda env: $ROOMCONN_CONDA_ENV" >&2
  echo "Tried env path: $ROOMCONN_CONDA_ENV_PATH" >&2
  echo "Set ROOMCONN_CONDA_ENV_PATH=/full/path/to/env if your env lives elsewhere." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

echo "[env] python: $(which python)"
python - <<'PY'
import sys
print("[env] version:", sys.version.split()[0])
try:
    import torch
    print("[env] torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
except Exception as e:
    print("[env] torch check failed:", repr(e))
PY

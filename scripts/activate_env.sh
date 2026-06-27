#!/bin/bash
# Shared environment activation for local cluster jobs.
#
# Default: Matthias' bwUniCluster conda environment.
# Override without editing scripts:
#   export ROOMCONN_CONDA_ENV=other_env_name_or_/path/to/env

set -e

ROOMCONN_CONDA_ENV="${ROOMCONN_CONDA_ENV:-roomconn}"
ROOMCONN_CONDA_ENV_PATH="${ROOMCONN_CONDA_ENV_PATH:-$HOME/.conda/envs/$ROOMCONN_CONDA_ENV}"

_roomconn_fail() {
  echo "ERROR: Could not activate conda env: $ROOMCONN_CONDA_ENV" >&2
  echo "Tried env path: $ROOMCONN_CONDA_ENV_PATH" >&2
  echo "Find the real path with: module load devel/miniforge && conda info --envs" >&2
  echo "Then set: export ROOMCONN_CONDA_ENV_PATH=/full/path/to/roomconn" >&2

  # If this file was sourced in an interactive shell, `exit` would close the SSH
  # session. Return when sourced; exit only when executed as a standalone script.
  if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 1
  else
    exit 1
  fi
}

# In SLURM batch jobs on bwUniCluster, the `conda` command can exist but fail with
# "ModuleNotFoundError: No module named 'conda'" on some nodes. Also, conda envs do
# not always provide a venv-style `$ENV/bin/activate`. The most robust path for our
# jobs is therefore to select the env's Python directly via PATH.
if [ -x "$ROOMCONN_CONDA_ENV_PATH/bin/python" ]; then
  export CONDA_PREFIX="$ROOMCONN_CONDA_ENV_PATH"
  export CONDA_DEFAULT_ENV="$ROOMCONN_CONDA_ENV"
  export PATH="$ROOMCONN_CONDA_ENV_PATH/bin:$PATH"
  unset PYTHONHOME
elif [ -d "$ROOMCONN_CONDA_ENV" ] && [ -x "$ROOMCONN_CONDA_ENV/bin/python" ]; then
  export CONDA_PREFIX="$ROOMCONN_CONDA_ENV"
  export CONDA_DEFAULT_ENV="$(basename "$ROOMCONN_CONDA_ENV")"
  export PATH="$ROOMCONN_CONDA_ENV/bin:$PATH"
  unset PYTHONHOME
elif [ "${ROOMCONN_ALLOW_CONDA_FALLBACK:-0}" = "1" ] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ROOMCONN_CONDA_ENV"
else
  _roomconn_fail
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

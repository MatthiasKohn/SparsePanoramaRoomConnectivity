#!/bin/bash
# Shared environment activation for local cluster jobs.
#
# Default: Matthias' bwUniCluster conda environment.
# Override without editing scripts:
#   export ROOMCONN_CONDA_ENV=/path/to/other/env

set -e

ROOMCONN_CONDA_ENV="${ROOMCONN_CONDA_ENV:-/home/ul/ul_student/ul_fnm03/.conda/envs/venv}"

if [ -d "$ROOMCONN_CONDA_ENV" ]; then
  # Prefer conda activation when available, because it also restores conda-specific
  # environment variables. Fall back to the env's activate script for batch shells.
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$ROOMCONN_CONDA_ENV"
  elif [ -f "$ROOMCONN_CONDA_ENV/bin/activate" ]; then
    source "$ROOMCONN_CONDA_ENV/bin/activate"
  else
    echo "ERROR: Found env directory but no activate script: $ROOMCONN_CONDA_ENV" >&2
    exit 1
  fi
else
  echo "ERROR: Conda env not found: $ROOMCONN_CONDA_ENV" >&2
  echo "Set ROOMCONN_CONDA_ENV=/path/to/env if your env lives elsewhere." >&2
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

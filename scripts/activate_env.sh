#!/bin/bash
# Shared environment activation for local cluster jobs.
#
# Default: Matthias' bwUniCluster conda environment.
# Override without editing scripts:
#   export ROOMCONN_CONDA_ENV=other_env_name_or_/path/to/env

set -e

ROOMCONN_CONDA_ENV="${ROOMCONN_CONDA_ENV:-roomconn}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ROOMCONN_CONDA_ENV"
elif [ -d "$ROOMCONN_CONDA_ENV" ] && [ -f "$ROOMCONN_CONDA_ENV/bin/activate" ]; then
  source "$ROOMCONN_CONDA_ENV/bin/activate"
else
  echo "ERROR: Could not activate conda env: $ROOMCONN_CONDA_ENV" >&2
  echo "Try loading miniforge first, e.g. 'module load devel/miniforge'." >&2
  echo "Or set ROOMCONN_CONDA_ENV=/full/path/to/env." >&2
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

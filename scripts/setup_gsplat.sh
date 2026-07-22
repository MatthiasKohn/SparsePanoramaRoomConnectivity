#!/bin/bash
# ONE-TIME gsplat install (run on a LOGIN node — it needs internet). After this, the GPU
# job never touches the login node. gsplat ships as source and JIT-compiles its CUDA kernel
# on first import on a GPU node (local nvcc, no internet) — we pre-warm that below.
#
#   bash scripts/setup_gsplat.sh
#
set -eo pipefail
module purge
module load profile/deeplrn cineca-ai/4.3.0

FMODELS="${FMODELS:-/leonardo_work/EUHPC_D35_121/envs/fmodels}"   # torch 2.5.0+cu124 venv
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/leonardo_work/EUHPC_D35_121/cache/torch_ext}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# install the gsplat python package (source). -u LD_LIBRARY_PATH avoids the cineca nvJitLink shadow.
env -u PYTHONPATH -u LD_LIBRARY_PATH "$FMODELS/bin/python" -m pip install --no-input gsplat

echo "gsplat installed. Version:"
env -u PYTHONPATH -u LD_LIBRARY_PATH "$FMODELS/bin/python" -c "import gsplat; print(gsplat.__version__)"
echo "NOTE: the CUDA kernel JIT-compiles on first GPU import (~a few min). The GPU job below"
echo "sets TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR so that compile is cached and reused."

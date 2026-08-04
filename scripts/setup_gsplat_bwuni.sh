#!/bin/bash
# Build gsplat's CUDA kernels for roomconn on bwUniCluster (login node; nvcc compile, no GPU needed).
#   bash scripts/setup_gsplat_bwuni.sh
#
# Why not `module load devel/cuda/12.8`? roomconn's torch is CUDA 13 (nvidia-*-cu13 wheels), and the
# cluster's newest CUDA module is 12.8 -> PyTorch refuses to build extensions across a CUDA major gap.
# So we build with a pip-provided nvcc 13 (matches torch) + a CUDA_HOME assembled from the cu13 wheels.
set -eo pipefail
EXT_ROOT="${EXT_ROOT:-/home/ul/ul_student/ul_fnm03/ext}"
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate roomconn

# nvcc 13 + CUDA 13 headers via pip. NOTE: the '-cu13'-suffixed names are deprecated redirect stubs
# that fail to build; the real packages dropped the suffix and are already 13.x to match roomconn's torch.
pip install --no-input nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-cccl

# assemble a unified CUDA_HOME (bin/include/lib64) by symlinking the pip nvidia CUDA packages
CUDA_HOME="$EXT_ROOT/cuda_home"; rm -rf "$CUDA_HOME"; mkdir -p "$CUDA_HOME/bin" "$CUDA_HOME/include" "$CUDA_HOME/lib64"
python - "$CUDA_HOME" <<'PY'
import sys, pathlib, nvidia
home = pathlib.Path(sys.argv[1]); base = pathlib.Path(nvidia.__path__[0])
for sub in base.iterdir():
    for kind, dst in [("bin", "bin"), ("include", "include"), ("lib", "lib64")]:
        s = sub / kind
        if s.is_dir():
            for f in s.iterdir():
                d = home / dst / f.name
                if not d.exists():
                    try: d.symlink_to(f)
                    except OSError: pass
print("assembled CUDA_HOME at", home)
PY
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0            # A100 = sm_80
nvcc --version || { echo "ERROR: nvcc not found in assembled CUDA_HOME"; ls "$CUDA_HOME/bin"; exit 1; }

# rebuild gsplat so its CUDA extension (_C) actually compiles
pip install --no-input --force-reinstall --no-build-isolation --no-cache-dir gsplat==1.5.3

# verify a compiled .so landed (can't run CUDA on the login node, but the built kernel is what we need)
python - <<'PY'
import gsplat, glob, os
sos = glob.glob(os.path.join(os.path.dirname(gsplat.__file__), "**", "*.so"), recursive=True)
print("gsplat .so files:", sos)
assert sos, "no compiled gsplat .so -- build failed; see errors above"
print("gsplat CUDA extension built OK")
PY
echo "DONE. run_floor_bwuni.slurm should now render (torch bundles the cu13 runtime; _C is prebuilt)."
echo "If this failed, fallback: a dedicated env with matched versions ->"
echo "  module load devel/cuda/12.8; mamba create -p $EXT_ROOT/envs/gsrender python=3.11;"
echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128; pip install gsplat lpips ..."

#!/bin/bash
# Install the floor-pipeline render deps into the roomconn env on bwUniCluster, IF missing.
# First just check (run_floor_bwuni.slurm prints this too):
#   module load devel/miniforge && conda activate roomconn
#   python -c "import torch,gsplat,lpips;print('ok',torch.__version__,torch.cuda.is_available())"
# If that errors on gsplat/lpips, run this on the login node:
#   bash scripts/setup_deps_bwuni.sh
set -eo pipefail
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate roomconn

# lpips (+ warm-cache AlexNet weights into torch hub so GPU nodes work offline)
pip install --no-input lpips
python - <<'PY'
import lpips; lpips.LPIPS(net="alex"); print("lpips + alexnet cached")
PY

# gsplat needs a CUDA toolkit to build its kernels. Adjust the module name if this one is absent
# (list options: `module avail cuda`). A100 = sm_80.
module load devel/cuda/12.4 2>/dev/null || module load toolkit/nvidia-cuda 2>/dev/null || \
  echo "WARN: no CUDA module loaded -- 'module avail cuda' and set the right one, then re-run."
export TORCH_CUDA_ARCH_LIST="8.0"
pip install --no-input gsplat

python -c "import gsplat, lpips, torch; print('deps OK:', torch.__version__, 'cuda', torch.cuda.is_available())"

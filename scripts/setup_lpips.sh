#!/bin/bash
# ONE-TIME (login node, needs internet): install lpips into fmodels + warm-cache the AlexNet
# backbone weights so the OFFLINE compute nodes can compute LPIPS. Without this, floor.py logs
# "LPIPS unavailable" and every results.csv row gets lpips=nan.
#
#   bash scripts/setup_lpips.sh
#
# lpips.LPIPS(net="alex") needs exactly one downloaded file: torchvision's alexnet weights
# (the LPIPS linear-calibration layer ships inside the pip package). We cache it into $TORCH_HOME
# so the compute nodes (TORCH_HOME set in env_leonardo.sh) load it offline.
set -eo pipefail
module purge
module load profile/deeplrn cineca-ai/4.3.0
FMODELS="${FMODELS:-/leonardo_work/EUHPC_D35_121/envs/fmodels}"
export TORCH_HOME="${TORCH_HOME:-/leonardo_work/EUHPC_D35_121/cache/torch}"
mkdir -p "$TORCH_HOME/hub/checkpoints"

# install lpips (strip PYTHONPATH/LD_LIBRARY_PATH so pip uses the fmodels venv, not cineca's shadow)
env -u PYTHONPATH -u LD_LIBRARY_PATH "$FMODELS/bin/python" -m pip install --no-input lpips

echo "warming the AlexNet backbone into $TORCH_HOME ..."
GCCLIB="$(dirname "$(g++ -print-file-name=libstdc++.so.6 2>/dev/null)")"; [[ "$GCCLIB" == /* ]] || GCCLIB=""
# HF_HUB_OFFLINE=0 so torchvision may fetch; import torch from fmodels (not cineca) via -u PYTHONPATH.
HF_HUB_OFFLINE=0 env -u PYTHONPATH LD_LIBRARY_PATH="$GCCLIB" TORCH_HOME="$TORCH_HOME" \
    "$FMODELS/bin/python" - <<'PY'
import lpips
m = lpips.LPIPS(net="alex")   # triggers the one alexnet-weights download into $TORCH_HOME
print("lpips ready; AlexNet weights cached. params:", sum(p.numel() for p in m.parameters()))
PY
echo "Done. Next floor.py runs will fill the lpips column (env_leonardo.sh already sets TORCH_HOME)."

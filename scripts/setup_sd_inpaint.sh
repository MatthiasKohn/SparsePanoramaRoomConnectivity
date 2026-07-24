#!/bin/bash
# ONE-TIME (login node, needs internet): install diffusers + cache the Stable-Diffusion
# inpainting weights so the offline compute nodes can load them. After this, run the oracle
# with `--inpaint sd` for generative hole-filling (default is 'cv2', classical, no setup).
#
#   bash scripts/setup_sd_inpaint.sh
#
set -eo pipefail
module purge
module load profile/deeplrn cineca-ai/4.3.0
FMODELS="${FMODELS:-/leonardo_work/EUHPC_D35_121/envs/fmodels}"
export HF_HOME="${HF_HOME:-/leonardo_work/EUHPC_D35_121/cache/huggingface}"
mkdir -p "$HF_HOME"

env -u PYTHONPATH -u LD_LIBRARY_PATH "$FMODELS/bin/python" -m pip install --no-input \
    diffusers transformers accelerate safetensors

echo "warming the SD-2-inpainting cache into $HF_HOME ..."
# import torch from the fmodels venv, NOT cineca's: strip PYTHONPATH (cineca torch shadow) and
# point LD_LIBRARY_PATH at only gcc-12 libstdc++ (no cineca CUDA shadow) — same fix as the runs.
GCCLIB="$(dirname "$(g++ -print-file-name=libstdc++.so.6 2>/dev/null)")"; [[ "$GCCLIB" == /* ]] || GCCLIB=""
# pass a VALID HF_TOKEN through (this model needs login). Set it first:  export HF_TOKEN=hf_...
: "${HF_TOKEN:?Set a valid token first — run:  export HF_TOKEN=hf_yourNewToken  (and accept the model card at huggingface.co/stabilityai/stable-diffusion-2-inpainting)}"
HF_HUB_OFFLINE=0 env -u PYTHONPATH LD_LIBRARY_PATH="$GCCLIB" HF_HOME="$HF_HOME" \
    HF_TOKEN="$HF_TOKEN" "$FMODELS/bin/python" - <<'PY'

import torch
from diffusers import StableDiffusionInpaintPipeline
StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
print("SD-2-inpainting cached OK")
PY
echo "Done. Run generative completion with:  (EXTRA='--inpaint sd' in run_oracle_floor.slurm)"
echo "env_leonardo.sh already sets HF_HOME + HF_HUB_OFFLINE=1 so compute nodes load from cache."

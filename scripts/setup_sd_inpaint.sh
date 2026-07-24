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
HF_HUB_OFFLINE=0 env -u LD_LIBRARY_PATH "$FMODELS/bin/python" - <<'PY'

import torch
from diffusers import StableDiffusionInpaintPipeline
StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
print("SD-2-inpainting cached OK")
PY
echo "Done. Run generative completion with:  (EXTRA='--inpaint sd' in run_oracle_floor.slurm)"
echo "env_leonardo.sh already sets HF_HOME + HF_HUB_OFFLINE=1 so compute nodes load from cache."

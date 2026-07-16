#!/bin/bash
# Run this ONCE on a LEONARDO LOGIN NODE (has internet). Compute nodes are OFFLINE, so we
# clone repos, install each model in its OWN venv, fetch weights, and warm the HF/torch caches
# here. After this, the compute-node job (probe_leonardo.slurm) runs fully offline.
#
#   bash overlap_probe/slurm/stage_models_login.sh
#
# Repos:
#   Argus     https://github.com/realsee-developer/RealSee3D   (weights: HF RealseeTechnology/argus-realsee3d)
#   PanoVGGT  https://github.com/YijingGuo-June/PanoVGGT        (weights on HF, see its README)
#   VGGT      https://github.com/facebookresearch/vggt          (weights auto-pull via HF)
set -euo pipefail

module purge; module load profile/deeplrn; module load cineca-ai/4.3.0
MODELS=/leonardo_work/EUHPC_D35_121/models
export HF_HOME=/leonardo_work/EUHPC_D35_121/cache/huggingface
export TORCH_HOME=/leonardo_work/EUHPC_D35_121/cache/torch
mkdir -p "$MODELS" "$HF_HOME" "$TORCH_HOME"

clone_and_env () {  # $1=name  $2=git-url
  local name="$1" url="$2" dir="$MODELS/$1"
  [ -d "$dir" ] || git clone "$url" "$dir"
  python -m venv "$dir/.venv"
  # shellcheck disable=SC1091
  source "$dir/.venv/bin/activate"
  pip install --upgrade pip
  pip install -r "$dir/requirements.txt" || echo "check $name README for install"
  deactivate
  echo "== $name ready at $dir  (python: $dir/.venv/bin/python) =="
}

clone_and_env argus    https://github.com/realsee-developer/RealSee3D
clone_and_env panovggt https://github.com/YijingGuo-June/PanoVGGT
clone_and_env vggt      https://github.com/facebookresearch/vggt

# Fetch published weights (edit filenames per each repo's README):
#   huggingface-cli download RealseeTechnology/argus-realsee3d --local-dir "$MODELS/argus/weights"
#   huggingface-cli download <panovggt-hf-id>                  --local-dir "$MODELS/panovggt/weights"

# WARM THE CACHES: run ONE demo image per model NOW so any hub/HF pull is cached offline.
#   source "$MODELS/vggt/.venv/bin/activate"; python "$MODELS/vggt/<demo>.py" ... ; deactivate
#   (repeat for argus / panovggt)

echo
echo "Next: (1) wire each adapter _cmd() in overlap_probe/adapters.py to the repo's real demo,"
echo "      (2) confirm each writes out/pred.npz['poses'] (N,4,4) c2w,"
echo "      (3) sbatch overlap_probe/slurm/probe_leonardo.slurm"

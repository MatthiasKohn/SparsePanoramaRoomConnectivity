#!/bin/bash
# ONE-TIME (login node, needs internet): set up the OFFICIAL SALVe pipeline so we can produce real
# estimated poses on ZInD floors, to compare against our GT oracle. Faithful integration -- uses
# SALVe's released verifier weights + released MHNet predictions + HoHoNet depth (trains nothing).
#
#   bash scripts/setup_salve.sh
#
# This is the heavy step. The conda env (GTSAM/GTSFM/Open3D) is the main risk; if the solve fails,
# see the fallback note at the bottom. Everything downloaded here is staged so the OFFLINE compute
# nodes can run without internet.
set -eo pipefail

# ---- where everything goes (on the shared work filesystem, not $HOME) ----
SALVE_ROOT="${SALVE_ROOT:-/leonardo_work/EUHPC_D35_121/ext/salve}"
HOHO_ROOT="${HOHO_ROOT:-/leonardo_work/EUHPC_D35_121/ext/HoHoNet}"
SALVE_ASSETS="${SALVE_ASSETS:-/leonardo_work/EUHPC_D35_121/ext/salve_assets}"
CONDA_ROOT="${CONDA_ROOT:-/leonardo_work/EUHPC_D35_121/ext/miniconda3}"
mkdir -p "$(dirname "$SALVE_ROOT")" "$SALVE_ASSETS/models" "$SALVE_ASSETS/mhnet_preds"

S3="https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve"

echo "=================================================================="
echo " 1/5  clone SALVe + HoHoNet"
echo "=================================================================="
[ -d "$SALVE_ROOT/.git" ] || git clone https://github.com/zillow/salve.git "$SALVE_ROOT"
[ -d "$HOHO_ROOT/.git" ]  || git clone https://github.com/sunset1995/HoHoNet.git "$HOHO_ROOT"

echo "=================================================================="
echo " 2/5  miniconda + SALVe conda env  (the risky part)"
echo "=================================================================="
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$CONDA_ROOT"
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
# The classic conda solver OOM-gets-Killed on the login-node cgroup for this heavy env. Solve with
# micromamba instead (tiny static binary, low memory, strict channel priority by default).
export MAMBA_ROOT_PREFIX="$CONDA_ROOT"
MM="$CONDA_ROOT/bin/micromamba"
if [ ! -x "$MM" ]; then
  wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$CONDA_ROOT" bin/micromamba
fi
# CURATED env. The full 2022 yml is unsolvable within the login-node memory cap (strict priority
# OOM-safe but hits an opencv conflict; flexible resolves it but OOM-Killed). SALVe's compiled deps
# (gtsam, gtsfm) are pip packages -> use a TRIVIAL conda env (python+pip: instant, tiny memory) and
# install everything with pip (opencv via pip too, avoiding the strict-priority opencv conflict).
if [ ! -x "$CONDA_ROOT/envs/salve-v1/bin/python" ]; then
  rm -rf "$CONDA_ROOT/envs/salve-v1"
  "$MM" create -y -c conda-forge -p "$CONDA_ROOT/envs/salve-v1" python=3.8 pip
fi
conda activate "$CONDA_ROOT/envs/salve-v1"
# era-appropriate torch for cudatoolkit 11.3 (A100/sm_80 supported); wheel bundles its CUDA runtime
pip install --no-input torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
# SALVe + HoHoNet deps (versions matched to their env file where they matter)
pip install --no-input \
    gtsam==4.2a7 gtsfm==0.2.0 "hydra-core==1.1.0" rdp yacs open3d "networkx>=2.6.3" \
    opencv-python "matplotlib>=3.4.2" numpy pandas "pillow>=8.0.1" scikit-learn seaborn \
    shapely tqdm click h5py imageio scipy simplejson colour pytest-cov
pip install --no-input -e "$SALVE_ROOT"


echo "=================================================================="
echo " 3/5  HoHoNet monodepth checkpoint  (SALVe's gdrive ID is dead -> fetch from HoHoNet's zoo)"
echo "=================================================================="
CKPT_DIR="$HOHO_ROOT/ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet"
if [ ! -s "$CKPT_DIR/ep60.pth" ]; then
  mkdir -p "$HOHO_ROOT/ckpt"
  # (a) Dropbox folder zip (dl=1) -- no Google-Drive quota limits
  wget -O /tmp/hoho_ckpt.zip "https://www.dropbox.com/sh/b014nop5jrehpoq/AACWNTMMHEAbaKOO1drqGio4a?dl=1" || true
  [ -s /tmp/hoho_ckpt.zip ] && unzip -o /tmp/hoho_ckpt.zip -d "$HOHO_ROOT/ckpt" || true
  # (b) fallback: gdown the Google-Drive folder
  if [ ! -s "$CKPT_DIR/ep60.pth" ]; then
    pip install --no-input gdown
    gdown --folder "https://drive.google.com/drive/folders/1raT3vRXnQXRAQuYq36dE-93xFc_hgkTQ" -O "$HOHO_ROOT/ckpt" || true
  fi
  # normalize: move the depth ckpt to where SALVe expects it, wherever it unzipped to
  found=$(find "$HOHO_ROOT/ckpt" -name ep60.pth -path '*mp3d_depth*' 2>/dev/null | head -1)
  if [ -n "$found" ] && [ ! -s "$CKPT_DIR/ep60.pth" ]; then mkdir -p "$CKPT_DIR"; cp "$found" "$CKPT_DIR/ep60.pth"; fi
fi
ls -lh "$CKPT_DIR/ep60.pth" && echo "HoHoNet ckpt OK" \
  || { echo "HoHoNet depth ckpt missing -- grab the 'mp3d_depth_...' ep60.pth manually from"; \
       echo "  https://www.dropbox.com/sh/b014nop5jrehpoq/AACWNTMMHEAbaKOO1drqGio4a?dl=0"; exit 1; }

echo "=================================================================="
echo " 4/5  SALVe verifier checkpoints (released) + MHNet predictions"
echo "=================================================================="
# realistic front-end (MHNet W/D/O): ResNet-152 ceiling+floor RGB, 587 tours
wget -nc -O "$SALVE_ASSETS/models/mhnet_ceiling_floor_587.pth"  "$S3/models/9fcbb628bd5efffbdcc4ce55a9eb380d.pth"
# clean front-end (GT W/D/O + GT layout): ResNet-152 ceiling+floor RGB, 817 tours
wget -nc -O "$SALVE_ASSETS/models/gtwdo_ceiling_floor_817.pth" "$S3/models/b1198bad27aecb8a19f884abc920a731.pth"
# MHNet predicted W/D/O + layout on all of ZInD (so we don't need the unreleased layout net)
if [ ! -e "$SALVE_ASSETS/mhnet_preds/.done" ]; then
  wget -nc -O "$SALVE_ASSETS/ZInD_HorizonNet_predictions.tar.gz" "$S3/data/ZInD_HorizonNet_predictions.tar.gz"
  tar -xzf "$SALVE_ASSETS/ZInD_HorizonNet_predictions.tar.gz" -C "$SALVE_ASSETS/mhnet_preds"
  touch "$SALVE_ASSETS/mhnet_preds/.done"
fi

echo "=================================================================="
echo " 5/5  vanishing-angle files (shipped in the repo, just split them)"
echo "=================================================================="
python "$SALVE_ROOT/scripts/split_vanishing_angle_file.py" \
    --csv "$SALVE_ROOT/assets/zind_vanishing_angles.csv" \
    --out "$SALVE_ASSETS/vanishing_angle"

echo
echo "=== available verifier YAML configs (pick the one matching the ckpt for run_salve.slurm) ==="
ls "$SALVE_ROOT/salve/configs/"
echo
echo "DONE. Paths for run_salve.slurm:"
echo "  SALVE_ROOT=$SALVE_ROOT   HOHO_ROOT=$HOHO_ROOT   SALVE_ASSETS=$SALVE_ASSETS   CONDA_ROOT=$CONDA_ROOT"
echo
echo "If a pip dep conflicts (e.g. gtsfm==0.2.0 vs numpy): relax that one pin and re-run; the conda"
echo "env itself is trivial (python+pip) so it won't OOM. Verify import: "
echo "  $CONDA_ROOT/envs/salve-v1/bin/python -c 'import salve, gtsam, gtsfm, cv2, open3d, torch; print(torch.cuda.is_available())'"

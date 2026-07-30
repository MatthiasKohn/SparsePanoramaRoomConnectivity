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
if ! conda env list | grep -q "salve-v1"; then
  # their pinned Linux env (brings GTSAM, GTSFM, Open3D, hydra, rdp, pytorch)
  conda env create -f "$SALVE_ROOT/environment_ubuntu-latest.yml"
fi
conda activate salve-v1
pip install -e "$SALVE_ROOT"


echo "=================================================================="
echo " 3/5  HoHoNet monodepth checkpoint"
echo "=================================================================="
( cd "$HOHO_ROOT" && bash "$SALVE_ROOT/scripts/download_monodepth_model.sh" )
ls -lh "$HOHO_ROOT/ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth" \
  && echo "HoHoNet ckpt OK"

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
echo "If the conda solve failed: the two brittle deps are GTSAM and GTSFM. Fallback is a manual env:"
echo "  conda create -n salve-v1 python=3.8 && conda activate salve-v1 && pip install -e $SALVE_ROOT \\"
echo "    && pip install gtsam gtsfm hydra-core rdp open3d && (retry, pin versions from the .yml as needed)"

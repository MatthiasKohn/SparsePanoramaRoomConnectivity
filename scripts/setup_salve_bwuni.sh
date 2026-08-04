#!/bin/bash
# ONE-TIME (bwUniCluster login node, needs internet): set up the OFFICIAL SALVe pipeline
# (LAYOUT-only verifier -> no depth/HoHoNet). bwUniCluster port of setup_salve.sh.
#   bash scripts/setup_salve_bwuni.sh
#
# bwUniCluster has miniforge as a module, so no miniconda bootstrap and no login-node OOM.
# NOTE on storage: $HOME has a small quota. If EXT_ROOT below fills it, allocate a workspace:
#   ws_allocate salve 60   ->  set EXT_ROOT to the returned /pfs/.../salve path and re-run.
set -eo pipefail

# ---- where everything goes (edit if you use a workspace instead of home) ----
EXT_ROOT="${EXT_ROOT:-/home/ul/ul_student/ul_fnm03/ext}"
SALVE_ROOT="${SALVE_ROOT:-$EXT_ROOT/salve}"
SALVE_ASSETS="${SALVE_ASSETS:-$EXT_ROOT/salve_assets}"
ENV_PREFIX="${ENV_PREFIX:-$EXT_ROOT/envs/salve-v1}"
mkdir -p "$EXT_ROOT" "$SALVE_ASSETS/models" "$SALVE_ASSETS/mhnet_preds"

S3="https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve"

echo "=== 1/4  clone SALVe ==="
[ -d "$SALVE_ROOT/.git" ] || git clone https://github.com/zillow/salve.git "$SALVE_ROOT"

echo "=== 2/4  conda env (miniforge module) + pip deps ==="
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  # trivial bare env (python+pip) -> instant, tiny; everything else via pip (like the Leonardo port)
  conda create -y -p "$ENV_PREFIX" python=3.8 pip
fi
conda activate "$ENV_PREFIX"
# era-appropriate torch for A100 (cu113 supports sm_80); wheel bundles its CUDA runtime
pip install --no-input torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install --no-input \
    gtsam==4.2a7 gtsfm==0.2.0 "hydra-core==1.1.0" rdp yacs open3d "networkx>=2.6.3" \
    opencv-python "matplotlib>=3.4.2" numpy pandas "pillow>=8.0.1" scikit-learn seaborn \
    shapely tqdm click h5py imageio scipy simplejson colour pytest-cov
pip install --no-input -e "$SALVE_ROOT"

echo "=== 3/4  SALVe verifier checkpoints + MHNet predictions (released) ==="
wget -nc -O "$SALVE_ASSETS/models/mhnet_layout_floor_877.pth" "$S3/models/6ac3f3e5fe6fa3d4bfae7c124d7787b3.pth"   # LAYOUT verifier (no depth)
wget -nc -O "$SALVE_ASSETS/models/mhnet_ceiling_floor_587.pth" "$S3/models/9fcbb628bd5efffbdcc4ce55a9eb380d.pth"  # (rgb, unused unless HoHoNet)
if [ ! -e "$SALVE_ASSETS/mhnet_preds/.done" ]; then
  wget -nc -O "$SALVE_ASSETS/ZInD_HorizonNet_predictions.tar.gz" "$S3/data/ZInD_HorizonNet_predictions.tar.gz"
  tar -xzf "$SALVE_ASSETS/ZInD_HorizonNet_predictions.tar.gz" -C "$SALVE_ASSETS/mhnet_preds"
  touch "$SALVE_ASSETS/mhnet_preds/.done"
fi

echo "=== 4/4  vanishing angles -> INSIDE the predictions root (parent of horizon_net/) ==="
HN=$(find "$SALVE_ASSETS/mhnet_preds" -maxdepth 4 -type d -name horizon_net 2>/dev/null | head -1)
[ -n "$HN" ] || { echo "ERROR: horizon_net/ not found -- check tar extraction:"; find "$SALVE_ASSETS/mhnet_preds" -maxdepth 2 | head; exit 1; }
PRED_ROOT=$(dirname "$HN")
python "$SALVE_ROOT/scripts/split_vanishing_angle_file.py" \
    --csv "$SALVE_ROOT/assets/zind_vanishing_angles.csv" --out "$PRED_ROOT/vanishing_angle"

echo
echo "=== available layout config (should list 6ac3f3e5...yaml) ==="
ls "$SALVE_ROOT/salve/configs/" | grep 6ac3f3e5 || ls "$SALVE_ROOT/salve/configs/"
echo "DONE. For run_salve_bwuni.slurm:"
echo "  SALVE_ROOT=$SALVE_ROOT   SALVE_ASSETS=$SALVE_ASSETS   ENV_PREFIX=$ENV_PREFIX   PRED_ROOT=$PRED_ROOT"
echo "Verify: $ENV_PREFIX/bin/python -c 'import salve,gtsam,gtsfm,cv2,open3d,torch;print(torch.__version__)'"

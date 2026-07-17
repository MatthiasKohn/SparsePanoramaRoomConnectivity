#!/bin/bash
# Run COLMAP 4.1.0 NATIVE panorama (spherical) SfM on one home's panos and drop the sparse
# model where exp27 --model expects it. Native mode registers the PANORAMAS directly, so the
# output image names ARE the pano stems -> exp24.load_colmap / exp27 consume it as-is (no
# tile rendering, no camera-convention gymnastics).
#
#   bash scripts/run_colmap_home.sh <HOME_DIR> [CAMERA_MODEL]
#
# e.g.  bash scripts/run_colmap_home.sh ../data/zind/full_dataset/0025
#
# CAMERA_MODEL defaults to SPHERE (COLMAP 4.1.0 spherical/equirectangular model). If your
# build names it differently, the feature_extractor error will list valid models -> pass it
# as the 2nd arg. Needs the `colmap` binary (4.1.0) on PATH.
set -e
cd "$(dirname "$0")/.."

HOME_DIR=${1:?"give the home dir, e.g. ../data/zind/full_dataset/0025"}
CAMERA_MODEL=${2:-SPHERE}
PANOS="$HOME_DIR/panos"
WORK="$HOME_DIR/colmap"
DB="$WORK/database.db"
SPARSE="$WORK/sparse"

[ -d "$PANOS" ] || { echo "no panos at $PANOS"; exit 1; }
command -v colmap >/dev/null 2>&1 || { echo "COLMAP not on PATH — install COLMAP 4.1.0."; exit 1; }
mkdir -p "$WORK" "$SPARSE"

echo "=========================================================="
echo " COLMAP panorama SfM   home: $(basename "$HOME_DIR")   model: $CAMERA_MODEL"
echo " panos: $(ls -1 "$PANOS"/*.jpg 2>/dev/null | wc -l)"
echo "=========================================================="

echo "[1/3] feature extraction (spherical camera, shared intrinsics)"
colmap feature_extractor --database_path "$DB" --image_path "$PANOS" \
    --ImageReader.camera_model "$CAMERA_MODEL" --ImageReader.single_camera 1

echo "[2/3] exhaustive matching"
colmap exhaustive_matcher --database_path "$DB"

echo "[3/3] incremental mapping"
colmap mapper --database_path "$DB" --image_path "$PANOS" --output_path "$SPARSE"

MODEL="$SPARSE/0"
if [ -d "$MODEL" ]; then
  echo "=========================================================="
  echo " done -> model at $MODEL"
  echo " registered images:"; ls "$MODEL" 2>/dev/null
  echo ""
  echo " feed to the hybrid pose graph:"
  echo "   python -m pipelines.hybrid_real --home $HOME_DIR \\"
  echo "       --depth_dir $HOME_DIR/dap_depth/depth_meters --floor floor_01 \\"
  echo "       --ckpt best.pt --model $MODEL"
  echo ""
  echo " and score coverage/pose vs GT:"
  echo "   python -m pipelines.colmap_compare --home $HOME_DIR --model $MODEL --floor floor_01"
  echo "=========================================================="
else
  echo "mapper produced no model at $MODEL — likely too little overlap between panos"
  echo "(that itself is the SALVe-style result: SfM can't bridge near-zero-overlap rooms)."
fi

#!/bin/bash
# One-command driver: depth generation -> real hybrid floor map for one ZInD home.
# (Optional) also draws the connectivity graph if an encoder ckpt is given.
#
#   bash scripts/run_hybrid_home.sh <HOME_ID> <ZIND_ROOT> [CKPT] [COLMAP_MODEL] [FLOOR]
#
# Examples:
#   bash scripts/run_hybrid_home.sh 0025 /home/ul/ul_student/ul_fnm03/data/zind/full_dataset
#   bash scripts/run_hybrid_home.sh 0025 .../full_dataset runs/hardneg/best.pt <colmap_dir> floor_01
#
# Depth needs the DAP repo beside the project (config.DAP_ROOT); GPU recommended.
set -e
cd "$(dirname "$0")/.."

HOME_ID=${1:?"give a home id, e.g. 0025"}
ZROOT=${2:?"give the ZInD full_dataset root"}
CKPT=${3:-}
MODEL=${4:-}
FLOOR=${5:-floor_01}

HOME_DIR="$ZROOT/$HOME_ID"
DEPTH_OUT="$HOME_DIR/dap_depth"
DEPTH_DIR="$DEPTH_OUT/depth_meters"
[ -d "$HOME_DIR/panos" ] || { echo "no panos at $HOME_DIR/panos"; exit 1; }

echo "=========================================================="
echo " home $HOME_ID  floor $FLOOR"
echo " ckpt:  ${CKPT:-<none, GT-oracle flip prior>}"
echo " colmap:${MODEL:-<none, emulated coverage via --colmap_frac>}"
echo "=========================================================="

# ---- 1. depth (skip if already there) ----
if [ -d "$DEPTH_DIR" ] && [ -n "$(ls -A "$DEPTH_DIR" 2>/dev/null)" ]; then
  echo "[1/3] depth already present in $DEPTH_DIR — skipping"
else
  echo "[1/3] generating DAP depth -> $DEPTH_DIR"
  python scripts/generate_depth.py --input_dir "$HOME_DIR/panos" \
      --output_dir "$DEPTH_OUT" --pattern "*.jpg"
fi
N=$(ls -1 "$DEPTH_DIR"/*.npy 2>/dev/null | wc -l)
echo "      depth maps: $N"
[ "$N" -gt 0 ] || { echo "depth generation produced nothing — check the DAP runner"; exit 1; }

# ---- 2. real hybrid floor map ----
echo "[2/3] hybrid floor map (door edges + COLMAP + flip prior -> pose graph)"
ARGS=(--home "$HOME_DIR" --depth_dir "$DEPTH_DIR" --floor "$FLOOR")
[ -n "$CKPT" ]  && ARGS+=(--ckpt "$CKPT")
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")
python -m pipelines.hybrid_real "${ARGS[@]}"

# ---- 3. connectivity graph (only if an encoder ckpt is given) ----
if [ -n "$CKPT" ]; then
  echo "[3/3] connectivity graph (assign scoring)"
  python -m pipelines.connectivity_graph --home "$HOME_DIR" --ckpt "$CKPT" --scoring assign
else
  echo "[3/3] skipped connectivity graph (no ckpt)"
fi

echo "=========================================================="
echo " done. outputs in results/hybrid/ and results/connectivity/"
echo "=========================================================="

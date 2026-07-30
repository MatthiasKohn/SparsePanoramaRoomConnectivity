#!/bin/bash
# Run the OFFICIAL SALVe pipeline (LAYOUT-only verifier -> no depth, no HoHoNet) on ONE building and
# export estimated poses in our --pose_file format. Prereq (once): bash scripts/setup_salve.sh
#   sbatch scripts/run_salve.slurm
#
# IMPORTANT: pick a building in SALVe's TEST split (the verifier was trained on train+val). 0000 is in
# TRAIN -> would be leaky. Test-split examples: 0021, 0203, 0990, 0809. Run floor.py's oracle on the
# SAME building for a fair head-to-head.
#
# Stages: (1) W/D/O alignment hypotheses [one building]  (2) rasterized-layout BEV render [no depth]
#         (3) SALVe layout verifier inference  (4) pose-graph optimization + export.
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=salve
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
SALVE_ROOT=/leonardo_work/EUHPC_D35_121/ext/salve
SALVE_ASSETS=/leonardo_work/EUHPC_D35_121/ext/salve_assets
CONDA_ROOT=/leonardo_work/EUHPC_D35_121/ext/miniconda3
ZIND_ROOT=/leonardo_work/EUHPC_D35_121/datasets/zind/full_dataset
OUT=/leonardo_work/EUHPC_D35_121/results/SparsePanoramaRoomConnectivity/results/salve_poses
BUILDING=0021                 # MUST be in SALVe's TEST split (see note above)
FLOOR=floor_01
SPLIT=test                    # split that contains $BUILDING (for the verifier dataloader)
VERIFIER_CKPT=$SALVE_ASSETS/models/mhnet_layout_floor_877.pth   # layout-only verifier
CONFIG_NAME=6ac3f3e5fe6fa3d4bfae7c124d7787b3.yaml               # its config (modalities=["layout"])
CONF=0.93
# =============================================================================

set -eo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/.."
REPO="$PWD"
mkdir -p logs "$OUT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate "$CONDA_ROOT/envs/salve-v1"
export SALVE_REPO_DIRPATH="$SALVE_ROOT"
HYP=$OUT/hypotheses; LAYOUT=$OUT/layout_bev; PREDS=$OUT/serialized_preds; DUMMY_DEPTH=$OUT/_nodepth
# MHNet predictions root = parent of the horizon_net/ dir (tar may nest it); vanishing_angle/ lives here too
MHNET=$(dirname "$(find "$SALVE_ASSETS/mhnet_preds" -maxdepth 4 -type d -name horizon_net 2>/dev/null | head -1)")
[ -d "$MHNET/horizon_net" ] || { echo "ERROR: horizon_net/ not found under $SALVE_ASSETS/mhnet_preds -- re-run setup_salve.sh"; exit 1; }
echo "MHNet predictions root: $MHNET"
mkdir -p "$DUMMY_DEPTH"

echo "### 1/4 W/D/O alignment hypotheses (building $BUILDING) ###"
PYTHONPATH="$SALVE_ROOT/scripts:$REPO:$PYTHONPATH" python salve_integration/gen_hypotheses_one_building.py \
    --building_id "$BUILDING" --raw_dataset_dir "$ZIND_ROOT" \
    --hypotheses_save_root "$HYP" --wdo_source horizon_net --mhnet_predictions_data_root "$MHNET"

echo "### 2/4 rasterized-layout BEV render (no depth) ###"
( cd "$SALVE_ROOT" && PYTHONPATH=./ python scripts/render_dataset_bev.py \
    --num_processes 8 --raw_dataset_dir "$ZIND_ROOT" --hypotheses_save_root "$HYP" \
    --depth_save_root "$DUMMY_DEPTH" --bev_save_root "$OUT/bev_unused" \
    --layout_save_root "$LAYOUT" --building_id "$BUILDING" )

echo "### 3/4 SALVe layout verifier inference ###"
# point the config's layout_data_root at our renders (SALVe expects you to edit this per README)
sed -i "s#^    layout_data_root:.*#    layout_data_root: $LAYOUT#" "$SALVE_ROOT/salve/configs/$CONFIG_NAME"
python "$SALVE_ROOT/scripts/test.py" \
    --gpu_ids 0 --model_ckpt_fpath "$VERIFIER_CKPT" \
    --config_name "$CONFIG_NAME" --serialization_save_dir "$PREDS" --split "$SPLIT"

echo "### 4/4 pose-graph optimization + export to our --pose_file format ###"
PYTHONPATH="$REPO:$PYTHONPATH" python salve_integration/export_salve_poses.py \
    --building_id "$BUILDING" --floor_id "$FLOOR" --raw_dataset_dir "$ZIND_ROOT" \
    --serialized_preds_json_dir "$PREDS" --hypotheses_save_root "$HYP" \
    --mhnet_predictions_data_root "$MHNET" --out "$OUT" --confidence_threshold "$CONF"

echo "=== done. Next (normal env): ==="
echo "  python -m pipelines.floor --home \$ZIND_ROOT/$BUILDING --floor $FLOOR --pose_model salve \\"
echo "      --pose_file $OUT/${BUILDING}_${FLOOR}_gt.json  --depth_model gt_layout   # convention check -> RMSE ~0"
echo "  then swap _gt.json -> _est.json for the real SALVe number, and run the oracle on $BUILDING too."
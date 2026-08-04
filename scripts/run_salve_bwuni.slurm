#!/bin/bash
# bwUniCluster port of run_salve.slurm (LAYOUT-only verifier -> no depth/HoHoNet).
# Prereq (once): bash scripts/setup_salve_bwuni.sh
#   sbatch scripts/run_salve_bwuni.slurm
# Use a TEST-split building (verifier trained on train+val): 0021, 0203, 0990, 0809, ...
# Stages: hypotheses(one building) -> rasterized-layout BEV(no depth) -> verifier -> pose-graph+export.
#SBATCH --job-name=salve
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
EXT_ROOT=/home/ul/ul_student/ul_fnm03/ext
SALVE_ROOT=$EXT_ROOT/salve
SALVE_ASSETS=$EXT_ROOT/salve_assets
ENV_PREFIX=$EXT_ROOT/envs/salve-v1
ZIND_ROOT=/home/ul/ul_student/ul_fnm03/data/zind/full_dataset
OUT=$EXT_ROOT/salve_poses
BUILDING=0021                 # MUST be in SALVe's TEST split
FLOOR=floor_01
SPLIT=test
VERIFIER_CKPT=$SALVE_ASSETS/models/mhnet_layout_floor_877.pth
CONFIG_NAME=6ac3f3e5fe6fa3d4bfae7c124d7787b3.yaml
CONF=0.93
# =============================================================================

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"; REPO="$PWD"
mkdir -p logs "$OUT"
module load devel/miniforge
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX" || source "$CONDA_BASE/bin/activate" "$ENV_PREFIX"
export SALVE_REPO_DIRPATH="$SALVE_ROOT"
HYP=$OUT/hypotheses; LAYOUT=$OUT/layout_bev; PREDS=$OUT/serialized_preds; DUMMY_DEPTH=$OUT/_nodepth
MHNET=$(dirname "$(find "$SALVE_ASSETS/mhnet_preds" -maxdepth 4 -type d -name horizon_net 2>/dev/null | head -1)")
[ -d "$MHNET/horizon_net" ] || { echo "ERROR: horizon_net/ not found -- re-run setup_salve_bwuni.sh"; exit 1; }
echo "MHNet predictions root: $MHNET"
if [ ! -e "$MHNET/vanishing_angle/${BUILDING}.json" ]; then
  python "$SALVE_ROOT/scripts/split_vanishing_angle_file.py" \
      --csv "$SALVE_ROOT/assets/zind_vanishing_angles.csv" --out "$MHNET/vanishing_angle"
fi
# --- patch SALVe repo bugs (reset to pristine first so patches are deterministic each run) ---
git -C "$SALVE_ROOT" checkout -- salve/dataset/hnet_prediction_loader.py scripts/render_dataset_bev.py 2>/dev/null || true
# (a) vanishing-angle JSON is keyed by pano FILENAME stem, but the loader indexed int i -> use the stem
sed -i 's#vanishing_angle_deg = vanishing_angles\[i\]#vanishing_angle_deg = vanishing_angles.get(Path(img_fpath).stem, 0.0)#' \
    "$SALVE_ROOT/salve/dataset/hnet_prediction_loader.py"
# (b) render_dataset_bev calls load_inferred_floor_pose_graphs() without predictions_data_root -> env default
sed -i 's#predictions_data_root: str$#predictions_data_root: str = os.environ.get("MHNET_PRED_ROOT")#' \
    "$SALVE_ROOT/salve/dataset/hnet_prediction_loader.py"
# (c) render_dataset_bev calls generate_texture_maps_for_pair() but never imports it -> add the import
sed -i '/^import salve.utils.bev_rendering_utils as bev_rendering_utils$/a from salve.utils.bev_rendering_utils import generate_texture_maps_for_pair' \
    "$SALVE_ROOT/scripts/render_dataset_bev.py"
export MHNET_PRED_ROOT="$MHNET"
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
sed -i "s#^    layout_data_root:.*#    layout_data_root: $LAYOUT#" "$SALVE_ROOT/salve/configs/$CONFIG_NAME"
python "$SALVE_ROOT/scripts/test.py" \
    --gpu_ids 0 --model_ckpt_fpath "$VERIFIER_CKPT" \
    --config_name "$CONFIG_NAME" --serialization_save_dir "$PREDS" --split "$SPLIT"

echo "### 4/4 pose-graph optimization + export to our --pose_file format ###"
PYTHONPATH="$REPO:$PYTHONPATH" python salve_integration/export_salve_poses.py \
    --building_id "$BUILDING" --floor_id "$FLOOR" --raw_dataset_dir "$ZIND_ROOT" \
    --serialized_preds_json_dir "$PREDS" --hypotheses_save_root "$HYP" \
    --mhnet_predictions_data_root "$MHNET" --out "$OUT" --confidence_threshold "$CONF"

echo "=== done. Then (roomconn env): floor.py --pose_model salve --pose_file $OUT/${BUILDING}_${FLOOR}_gt.json (RMSE~0 check) then _est.json ==="

#!/bin/bash
# SALVe RGB-texture verifier (the paper's strong modality) fed by OUR depth instead of dead HoHoNet.
# One building. Prereq (once): bash scripts/setup_salve_bwuni.sh
#   sbatch scripts/run_salve_rgb_bwuni.slurm
# Validate with DEPTH_MODEL=gt_layout first (clean depth), then dap/pager (monocular = your regime).
#SBATCH --job-name=salve_rgb
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
EXT_ROOT=/home/ul/ul_student/ul_fnm03/ext
SALVE_ROOT=$EXT_ROOT/salve
SALVE_ASSETS=$EXT_ROOT/salve_assets
ENV_PREFIX=$EXT_ROOT/envs/salve-v1
ZIND_ROOT=/home/ul/ul_student/ul_fnm03/data/zind/full_dataset
OUT=$EXT_ROOT/salve_poses_rgb
BUILDING=0021
FLOOR=floor_01
SPLIT=test
DEPTH_MODEL=gt_layout          # gt_layout (clean, validate first) | dap | pager (monocular)
VERIFIER_CKPT=$SALVE_ASSETS/models/mhnet_ceiling_floor_587.pth   # RGB verifier
CONFIG_NAME=9fcbb628bd5efffbdcc4ce55a9eb380d.yaml                # its config (ceiling+floor rgb)
CONF=0.5
# =============================================================================

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"; REPO="$PWD"; mkdir -p logs "$OUT"
module load devel/miniforge
CONDA_BASE="$(conda info --base)"; source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX" || source "$CONDA_BASE/bin/activate" "$ENV_PREFIX"
export SALVE_REPO_DIRPATH="$SALVE_ROOT" ZIND_ROOT RESULTS_ROOT="$REPO/results" PROJECT_ROOT="$REPO"
MHNET=$(dirname "$(find "$SALVE_ASSETS/mhnet_preds" -maxdepth 4 -type d -name horizon_net 2>/dev/null | head -1)")
export MHNET_PRED_ROOT="$MHNET"
[ -e "$MHNET/vanishing_angle/${BUILDING}.json" ] || python "$SALVE_ROOT/scripts/split_vanishing_angle_file.py" \
    --csv "$SALVE_ROOT/assets/zind_vanishing_angles.csv" --out "$MHNET/vanishing_angle"
# --- SALVe repo bug patches (same as layout run) ---
git -C "$SALVE_ROOT" checkout -- salve/dataset/hnet_prediction_loader.py scripts/render_dataset_bev.py salve/utils/io.py 2>/dev/null || true
sed -i 's#vanishing_angle_deg = vanishing_angles\[i\]#vanishing_angle_deg = vanishing_angles.get(Path(img_fpath).stem, 0.0)#' "$SALVE_ROOT/salve/dataset/hnet_prediction_loader.py"
sed -i 's#predictions_data_root: str$#predictions_data_root: str = os.environ.get("MHNET_PRED_ROOT")#' "$SALVE_ROOT/salve/dataset/hnet_prediction_loader.py"
sed -i '/^import salve.utils.bev_rendering_utils as bev_rendering_utils$/a from salve.utils.bev_rendering_utils import generate_texture_maps_for_pair' "$SALVE_ROOT/scripts/render_dataset_bev.py"
sed -i 's#os.makedirs(os.path.dirname(json_fpath), exist_ok=True)#os.makedirs(os.path.dirname(json_fpath) or ".", exist_ok=True)#' "$SALVE_ROOT/salve/utils/io.py"
sed -i 's#json.dump(data, f, indent=4)#json.dump(data, f, indent=4, default=str)#' "$SALVE_ROOT/salve/utils/io.py"

HYP=$OUT/$BUILDING/hyp; DEPTH=$OUT/$BUILDING/depth; BEV=$OUT/$BUILDING/bev; PREDS=$OUT/$BUILDING/preds

echo "### 0/5 write OUR depth ($DEPTH_MODEL) in SALVe format -> disables HoHoNet ###"
PYTHONPATH="$REPO" python salve_integration/make_salve_depth.py \
    --home "$ZIND_ROOT/$BUILDING" --building_id "$BUILDING" --depth_save_root "$DEPTH" --depth_model "$DEPTH_MODEL"

echo "### 1/5 W/D/O alignment hypotheses ###"
PYTHONPATH="$SALVE_ROOT/scripts:$REPO" python salve_integration/gen_hypotheses_one_building.py \
    --building_id "$BUILDING" --raw_dataset_dir "$ZIND_ROOT" --hypotheses_save_root "$HYP" \
    --wdo_source horizon_net --mhnet_predictions_data_root "$MHNET"

echo "### 2/5 RGB-texture BEV render (uses OUR depth; no --layout_save_root) ###"
( cd "$SALVE_ROOT" && PYTHONPATH=./ python scripts/render_dataset_bev.py --num_processes 8 \
    --raw_dataset_dir "$ZIND_ROOT" --hypotheses_save_root "$HYP" --depth_save_root "$DEPTH" \
    --bev_save_root "$BEV" --building_id "$BUILDING" )

echo "### 3/5 SALVe RGB verifier inference ###"
sed -i "s#^    data_root:.*#    data_root: $BEV#" "$SALVE_ROOT/salve/configs/$CONFIG_NAME"
python "$SALVE_ROOT/scripts/test.py" --gpu_ids 0 --model_ckpt_fpath "$VERIFIER_CKPT" \
    --config_name "$CONFIG_NAME" --serialization_save_dir "$PREDS" --split "$SPLIT"

echo "### 4/5 pose graph + export ###"
PYTHONPATH="$REPO" python salve_integration/export_salve_poses.py \
    --building_id "$BUILDING" --floor_id "$FLOOR" --raw_dataset_dir "$ZIND_ROOT" \
    --serialized_preds_json_dir "$PREDS" --hypotheses_save_root "$HYP" \
    --mhnet_predictions_data_root "$MHNET" --out "$OUT" --confidence_threshold "$CONF"

echo "### 5/5 metrics: acceptance (_gt ~0), real (_est), oracle ###"
PYTHONPATH="$REPO" python -m pipelines.floor --home "$ZIND_ROOT/$BUILDING" --floor "$FLOOR" \
    --pose_model salve --pose_file "$OUT/${BUILDING}_${FLOOR}_gt.json"  --tag salve_rgb_gtcheck --metrics_only
PYTHONPATH="$REPO" python -m pipelines.floor --home "$ZIND_ROOT/$BUILDING" --floor "$FLOOR" \
    --pose_model salve --pose_file "$OUT/${BUILDING}_${FLOOR}_est.json" --tag "salve_rgb_${DEPTH_MODEL}" --metrics_only
PYTHONPATH="$REPO" python -m pipelines.floor --home "$ZIND_ROOT/$BUILDING" --floor "$FLOOR" \
    --pose_model gt --tag oracle --metrics_only
echo "=== done. compare salve_rgb_${DEPTH_MODEL} vs the earlier layout-only salve_est and the oracle ==="

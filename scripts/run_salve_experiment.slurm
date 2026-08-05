#!/bin/bash
# REAL EXPERIMENT: real SALVe (layout-only) vs the GT oracle across many ZInD TEST-split floors.
# One submission -> for each floor: run SALVe (hypotheses->layout BEV->verifier->pose graph->export),
# then append 3 rows to results.csv: SALVe _gt (acceptance, must be ~0), SALVe _est (the real number),
# and the GT oracle. Produces a distribution of (pose_rmse, rot_err, door_gap, localization) to compare.
# Prereq (once): bash scripts/setup_salve_bwuni.sh
#   sbatch scripts/run_salve_experiment.slurm
#SBATCH --job-name=salve_exp
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
EXT_ROOT=/home/ul/ul_student/ul_fnm03/ext
SALVE_ROOT=$EXT_ROOT/salve
SALVE_ASSETS=$EXT_ROOT/salve_assets
ENV_PREFIX=$EXT_ROOT/envs/salve-v1
ZIND_ROOT=/home/ul/ul_student/ul_fnm03/data/zind/full_dataset
OUT=$EXT_ROOT/salve_poses
FLOOR=floor_01
SPLIT=test
VERIFIER_CKPT=$SALVE_ASSETS/models/mhnet_layout_floor_877.pth
CONFIG_NAME=6ac3f3e5fe6fa3d4bfae7c124d7787b3.yaml
CONF=0.5
# TEST-split buildings (skip 1348: duplicate pano IDs). Trim/extend as desired.
BUILDINGS=(0021 0203 0809 0966 1041 0668 0127 0494 0854 1184 0629 0576 0100 0190 0431)
# =============================================================================

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"; REPO="$PWD"
mkdir -p logs "$OUT"
module load devel/miniforge
CONDA_BASE="$(conda info --base)"; source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX" || source "$CONDA_BASE/bin/activate" "$ENV_PREFIX"
export SALVE_REPO_DIRPATH="$SALVE_ROOT"
export ZIND_ROOT RESULTS_ROOT="$REPO/results" PROJECT_ROOT="$REPO"
MHNET=$(dirname "$(find "$SALVE_ASSETS/mhnet_preds" -maxdepth 4 -type d -name horizon_net 2>/dev/null | head -1)")
[ -d "$MHNET/horizon_net" ] || { echo "ERROR: horizon_net/ not found -- re-run setup_salve_bwuni.sh"; exit 1; }
export MHNET_PRED_ROOT="$MHNET"
[ -e "$MHNET/vanishing_angle/0021.json" ] || python "$SALVE_ROOT/scripts/split_vanishing_angle_file.py" \
    --csv "$SALVE_ROOT/assets/zind_vanishing_angles.csv" --out "$MHNET/vanishing_angle"

# --- patch SALVe repo bugs (reset to pristine first, deterministic each run) ---
git -C "$SALVE_ROOT" checkout -- salve/dataset/hnet_prediction_loader.py scripts/render_dataset_bev.py salve/utils/io.py 2>/dev/null || true
sed -i 's#vanishing_angle_deg = vanishing_angles\[i\]#vanishing_angle_deg = vanishing_angles.get(Path(img_fpath).stem, 0.0)#' "$SALVE_ROOT/salve/dataset/hnet_prediction_loader.py"
sed -i 's#predictions_data_root: str$#predictions_data_root: str = os.environ.get("MHNET_PRED_ROOT")#' "$SALVE_ROOT/salve/dataset/hnet_prediction_loader.py"
sed -i '/^import salve.utils.bev_rendering_utils as bev_rendering_utils$/a from salve.utils.bev_rendering_utils import generate_texture_maps_for_pair' "$SALVE_ROOT/scripts/render_dataset_bev.py"
sed -i 's#os.makedirs(os.path.dirname(json_fpath), exist_ok=True)#os.makedirs(os.path.dirname(json_fpath) or ".", exist_ok=True)#' "$SALVE_ROOT/salve/utils/io.py"
sed -i 's#json.dump(data, f, indent=4)#json.dump(data, f, indent=4, default=str)#' "$SALVE_ROOT/salve/utils/io.py"

run_one() (            # subshell: a failure aborts THIS building only (caught by || below)
  set -eo pipefail
  B="$1"; OUTB="$OUT/exp/$B"; HYP="$OUTB/hyp"; LAYOUT="$OUTB/layout"; PREDS="$OUTB/preds"; DD="$OUTB/_nodepth"
  mkdir -p "$DD"
  [ -e "$MHNET/vanishing_angle/${B}.json" ] || python "$SALVE_ROOT/scripts/split_vanishing_angle_file.py" \
      --csv "$SALVE_ROOT/assets/zind_vanishing_angles.csv" --out "$MHNET/vanishing_angle"
  echo "  [1/4] hypotheses"
  PYTHONPATH="$SALVE_ROOT/scripts:$REPO" python salve_integration/gen_hypotheses_one_building.py \
      --building_id "$B" --raw_dataset_dir "$ZIND_ROOT" --hypotheses_save_root "$HYP" \
      --wdo_source horizon_net --mhnet_predictions_data_root "$MHNET"
  echo "  [2/4] layout BEV"
  ( cd "$SALVE_ROOT" && PYTHONPATH=./ python scripts/render_dataset_bev.py --num_processes 8 \
      --raw_dataset_dir "$ZIND_ROOT" --hypotheses_save_root "$HYP" --depth_save_root "$DD" \
      --bev_save_root "$OUTB/bev_unused" --layout_save_root "$LAYOUT" --building_id "$B" )
  echo "  [3/4] verifier"
  sed -i "s#^    layout_data_root:.*#    layout_data_root: $LAYOUT#" "$SALVE_ROOT/salve/configs/$CONFIG_NAME"
  python "$SALVE_ROOT/scripts/test.py" --gpu_ids 0 --model_ckpt_fpath "$VERIFIER_CKPT" \
      --config_name "$CONFIG_NAME" --serialization_save_dir "$PREDS" --split "$SPLIT"
  echo "  [4/4] pose graph + export"
  PYTHONPATH="$REPO" python salve_integration/export_salve_poses.py \
      --building_id "$B" --floor_id "$FLOOR" --raw_dataset_dir "$ZIND_ROOT" \
      --serialized_preds_json_dir "$PREDS" --hypotheses_save_root "$HYP" \
      --mhnet_predictions_data_root "$MHNET" --out "$OUT" --confidence_threshold "$CONF"
  echo "  [metrics] salve _gt (acceptance ~0), salve _est (real), gt oracle"
  PYTHONPATH="$REPO" python -m pipelines.floor --home "$ZIND_ROOT/$B" --floor "$FLOOR" \
      --pose_model salve --pose_file "$OUT/${B}_${FLOOR}_gt.json"  --tag salve_gtcheck --metrics_only
  PYTHONPATH="$REPO" python -m pipelines.floor --home "$ZIND_ROOT/$B" --floor "$FLOOR" \
      --pose_model salve --pose_file "$OUT/${B}_${FLOOR}_est.json" --tag salve_est --metrics_only
  PYTHONPATH="$REPO" python -m pipelines.floor --home "$ZIND_ROOT/$B" --floor "$FLOOR" \
      --pose_model gt --tag oracle --metrics_only
)

for B in "${BUILDINGS[@]}"; do
  echo; echo "########## building $B ##########"
  run_one "$B" || echo "  !! $B failed -- skipping"
done
echo; echo "=== experiment done. rows -> $RESULTS_ROOT/floor/results.csv ==="
column -s, -t "$RESULTS_ROOT/floor/results.csv" 2>/dev/null | grep -E "home|,salve,|,gt," || true

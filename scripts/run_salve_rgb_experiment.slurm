#!/bin/bash
# REAL EXPERIMENT: SALVe RGB-texture verifier (paper's strong modality) fed by OUR depth, vs the GT
# oracle, across many ZInD TEST-split floors. One submission -> per floor: write our depth -> SALVe
# (hypotheses->RGB BEV->RGB verifier->pose graph->export) -> 3 tagged rows in results.csv.
# Prereq (once): bash scripts/setup_salve_bwuni.sh
#   sbatch scripts/run_salve_rgb_experiment.slurm
#SBATCH --job-name=salve_rgb_exp
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
EXT_ROOT=/home/ul/ul_student/ul_fnm03/ext
SALVE_ROOT=$EXT_ROOT/salve
SALVE_ASSETS=$EXT_ROOT/salve_assets
ENV_PREFIX=$EXT_ROOT/envs/salve-v1
ZIND_ROOT=/home/ul/ul_student/ul_fnm03/data/zind/full_dataset
OUT=$EXT_ROOT/salve_poses_rgb
FLOOR=floor_01
SPLIT=test
DEPTH_MODEL=gt_layout          # gt_layout (clean) | dap | pager (monocular = your regime probe)
VERIFIER_CKPT=$SALVE_ASSETS/models/mhnet_ceiling_floor_587.pth
CONFIG_NAME=9fcbb628bd5efffbdcc4ce55a9eb380d.yaml
CONF=0.5
BUILDINGS=(0021 0203 0809 0966 1041 0668 0127 0494 0854 1184 0629 0576 0100 0190 0431)
# =============================================================================




set -eo pipefail
cd "$SLURM_SUBMIT_DIR"; REPO="$PWD"; mkdir -p logs "$OUT"
module load devel/miniforge
# Locate conda base by PATH (do NOT run the `conda` stub: it's broken on some bwHPC compute nodes,
# "No module named 'conda'"). Then activate best-effort, and force the env's python onto PATH so all
# `python` calls use it regardless of whether conda activation succeeds on this node.
CONDA_BASE="$(dirname "$(dirname "$(command -v conda)")")"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_PREFIX" 2>/dev/null || source "$CONDA_BASE/bin/activate" "$ENV_PREFIX" 2>/dev/null || true
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export SALVE_REPO_DIRPATH="$SALVE_ROOT" ZIND_ROOT RESULTS_ROOT="$REPO/results" PROJECT_ROOT="$REPO"
MHNET=$(dirname "$(find "$SALVE_ASSETS/mhnet_preds" -maxdepth 4 -type d -name horizon_net 2>/dev/null | head -1)")
export MHNET_PRED_ROOT="$MHNET"
[ -e "$MHNET/vanishing_angle/0021.json" ] || python "$SALVE_ROOT/scripts/split_vanishing_angle_file.py" \
    --csv "$SALVE_ROOT/assets/zind_vanishing_angles.csv" --out "$MHNET/vanishing_angle"
# --- SALVe repo bug patches (reset then apply; deterministic each run) ---
git -C "$SALVE_ROOT" checkout -- salve/dataset/hnet_prediction_loader.py scripts/render_dataset_bev.py salve/utils/io.py salve/utils/bev_rendering_utils.py salve/utils/hohonet_inference.py 2>/dev/null || true
sed -i 's#vanishing_angle_deg = vanishing_angles\[i\]#vanishing_angle_deg = vanishing_angles.get(Path(img_fpath).stem, 0.0)#' "$SALVE_ROOT/salve/dataset/hnet_prediction_loader.py"
sed -i 's#predictions_data_root: str$#predictions_data_root: str = os.environ.get("MHNET_PRED_ROOT")#' "$SALVE_ROOT/salve/dataset/hnet_prediction_loader.py"
sed -i '/^import salve.utils.bev_rendering_utils as bev_rendering_utils$/a from salve.utils.bev_rendering_utils import generate_texture_maps_for_pair' "$SALVE_ROOT/scripts/render_dataset_bev.py"
sed -i 's#os.makedirs(os.path.dirname(json_fpath), exist_ok=True)#os.makedirs(os.path.dirname(json_fpath) or ".", exist_ok=True)#' "$SALVE_ROOT/salve/utils/io.py"
sed -i 's#json.dump(data, f, indent=4)#json.dump(data, f, indent=4, default=str)#' "$SALVE_ROOT/salve/utils/io.py"
sed -i 's#bev_rendering_utils\.render_bev_pair#render_bev_pair#g' "$SALVE_ROOT/salve/utils/bev_rendering_utils.py"
# (g) if any pano's depth is missing SALVe calls broken HoHoNet (update_config) and kills the whole
#     render pool. Replace that fallback with a zero-depth write -> that pair renders empty & is skipped.
sed -i 's#^    infer_depth(args)#    import numpy as _np, imageio as _im; _im.imwrite(f"{building_depth_save_dir}/{fname_stem}.depth.png", _np.zeros((512,1024),_np.uint16)); return#' "$SALVE_ROOT/salve/utils/hohonet_inference.py"

run_one() (
  set -eo pipefail
  B="$1"; O="$OUT/$B"; HYP="$O/hyp"; DEPTH="$O/depth"; BEV="$O/bev"; PREDS="$O/preds"
  [ -e "$MHNET/vanishing_angle/${B}.json" ] || python "$SALVE_ROOT/scripts/split_vanishing_angle_file.py" \
      --csv "$SALVE_ROOT/assets/zind_vanishing_angles.csv" --out "$MHNET/vanishing_angle"
  echo "  [0] depth ($DEPTH_MODEL)"
  PYTHONPATH="$REPO" python salve_integration/make_salve_depth.py --home "$ZIND_ROOT/$B" \
      --building_id "$B" --depth_save_root "$DEPTH" --depth_model "$DEPTH_MODEL"
  echo "  [1] hypotheses"
  PYTHONPATH="$SALVE_ROOT/scripts:$REPO" python salve_integration/gen_hypotheses_one_building.py \
      --building_id "$B" --raw_dataset_dir "$ZIND_ROOT" --hypotheses_save_root "$HYP" \
      --wdo_source horizon_net --mhnet_predictions_data_root "$MHNET"
  echo "  [2] RGB BEV render (our depth)"
  ( cd "$SALVE_ROOT" && PYTHONPATH=./ python scripts/render_dataset_bev.py --num_processes 8 \
      --raw_dataset_dir "$ZIND_ROOT" --hypotheses_save_root "$HYP" --depth_save_root "$DEPTH" \
      --bev_save_root "$BEV" --building_id "$B" )
  echo "  [3] RGB verifier"
  sed -i "s#^    data_root:.*#    data_root: $BEV#" "$SALVE_ROOT/salve/configs/$CONFIG_NAME"
  python "$SALVE_ROOT/scripts/test.py" --gpu_ids 0 --model_ckpt_fpath "$VERIFIER_CKPT" \
      --config_name "$CONFIG_NAME" --serialization_save_dir "$PREDS" --split "$SPLIT"
  echo "  [4] pose graph + export"
  PYTHONPATH="$REPO" python salve_integration/export_salve_poses.py --building_id "$B" --floor_id "$FLOOR" \
      --raw_dataset_dir "$ZIND_ROOT" --serialized_preds_json_dir "$PREDS" --hypotheses_save_root "$HYP" \
      --mhnet_predictions_data_root "$MHNET" --out "$OUT" --confidence_threshold "$CONF"
  echo "  [5] metrics"
  PYTHONPATH="$REPO" python -m pipelines.floor --home "$ZIND_ROOT/$B" --floor "$FLOOR" \
      --pose_model salve --pose_file "$OUT/${B}_${FLOOR}_gt.json"  --tag salve_rgb_gtcheck --metrics_only
  PYTHONPATH="$REPO" python -m pipelines.floor --home "$ZIND_ROOT/$B" --floor "$FLOOR" \
      --pose_model salve --pose_file "$OUT/${B}_${FLOOR}_est.json" --tag "salve_rgb_${DEPTH_MODEL}" --metrics_only
  PYTHONPATH="$REPO" python -m pipelines.floor --home "$ZIND_ROOT/$B" --floor "$FLOOR" \
      --pose_model gt --tag oracle --metrics_only
)

for B in "${BUILDINGS[@]}"; do
  echo; echo "########## building $B ##########"
  run_one "$B" || echo "  !! $B failed -- skipping"
done
echo; echo "=== done. analyze: python scripts/analyze_salve_experiment.py --csv $RESULTS_ROOT/floor/results.csv --out $RESULTS_ROOT/floor ==="

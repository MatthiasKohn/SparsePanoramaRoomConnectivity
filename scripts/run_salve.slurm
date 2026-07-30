#!/bin/bash
# Run the OFFICIAL SALVe pipeline end-to-end and export estimated poses in our --pose_file format.
# Prereq (once): bash scripts/setup_salve.sh
#   sbatch scripts/run_salve.slurm
# Produces: $OUT/<building>_<floor>_est.json (+ _gt.json convention check). Then, in the normal pipeline:
#   python -m pipelines.floor --home $ZIND_ROOT/0000 --floor floor_01 \
#       --pose_model salve --pose_file $OUT/0000_floor_01_est.json --depth_model gt_layout
#
# NOTE: SALVe processes a whole ZInD SPLIT. Stages 1-4 over the test split take real GPU hours; that
# is how SALVe is designed. Use --building_id where a stage supports it to shrink the first run.
# Stage flags follow the official README; the FIRST run will likely surface a path/flag tweak or two.
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=salve
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
SALVE_ROOT=/leonardo_work/EUHPC_D35_121/ext/salve
HOHO_ROOT=/leonardo_work/EUHPC_D35_121/ext/HoHoNet
SALVE_ASSETS=/leonardo_work/EUHPC_D35_121/ext/salve_assets
CONDA_ROOT=/leonardo_work/EUHPC_D35_121/ext/miniconda3
ZIND_ROOT=/leonardo_work/EUHPC_D35_121/datasets/zind/full_dataset
OUT=/leonardo_work/EUHPC_D35_121/results/SparsePanoramaRoomConnectivity/results/salve_poses
SPLIT=test                                  # ZInD split to process (target building must be in it)
BUILDING=0000                               # building/floor to EXPORT poses for at the end
FLOOR=floor_01
WDO_SOURCE=horizon_net                       # horizon_net (realistic MHNet) | gt (GT W/D/O)
VERIFIER_CKPT=$SALVE_ASSETS/models/mhnet_ceiling_floor_587.pth   # match to WDO_SOURCE
# config must match the ckpt: 9fcbb628=mhnet_587 (horizon_net) ; b1198bad=gtwdo_817 (gt). Drop .yaml if test.py rejects it.
CONFIG_NAME=9fcbb628bd5efffbdcc4ce55a9eb380d.yaml
CONF=0.93
# =============================================================================

set -eo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/.."
mkdir -p logs "$OUT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate salve-v1
export SALVE_REPO_DIRPATH="$SALVE_ROOT"
HYP=$OUT/hypotheses; DEPTH=$OUT/depth; BEV=$OUT/bev; PREDS=$OUT/serialized_preds
MHNET=$SALVE_ASSETS/mhnet_preds

[ -z "$CONFIG_NAME" ] && { echo "SET CONFIG_NAME (see: ls $SALVE_ROOT/salve/configs/)"; exit 1; }

echo "### 1/5 alignment hypotheses (door/W-D-O snapping) ###"
python "$SALVE_ROOT/scripts/export_alignment_hypotheses.py" \
    --num_processes 16 --raw_dataset_dir "$ZIND_ROOT" --hypotheses_save_root "$HYP" \
    --wdo_source "$WDO_SOURCE" --split "$SPLIT" --mhnet_predictions_data_root "$MHNET"

echo "### 2/5 HoHoNet depth ###"
( cd "$HOHO_ROOT" && python "$SALVE_ROOT/scripts/batch_hohonet_inference.py" \
    --raw_dataset_dir "$ZIND_ROOT" --depth_save_root "$DEPTH" --num_processes 4 )

echo "### 3/5 BEV texture maps ###"
( cd "$SALVE_ROOT" && PYTHONPATH=./ python "$SALVE_ROOT/scripts/render_dataset_bev.py" \
    --num_processes 8 --raw_dataset_dir "$ZIND_ROOT" --hypotheses_save_root "$HYP" \
    --depth_save_root "$DEPTH" --bev_save_root "$BEV" --split "$SPLIT" )

echo "### 4/5 SALVe verifier inference (pretrained) ###"
python "$SALVE_ROOT/scripts/test.py" \
    --gpu_ids 0 --model_ckpt_fpath "$VERIFIER_CKPT" \
    --config_name "$CONFIG_NAME" --serialization_save_dir "$PREDS"

echo "### 5/5 pose-graph optimization + export to our --pose_file format ###"
python salve_integration/export_salve_poses.py \
    --building_id "$BUILDING" --floor_id "$FLOOR" --raw_dataset_dir "$ZIND_ROOT" \
    --serialized_preds_json_dir "$PREDS" --hypotheses_save_root "$HYP" \
    --mhnet_predictions_data_root "$MHNET" --out "$OUT" --confidence_threshold "$CONF"

echo "=== done. Next (normal env): floor.py --pose_model salve --pose_file $OUT/${BUILDING}_${FLOOR}_est.json ==="
echo "=== FIRST verify convention: --pose_file $OUT/${BUILDING}_${FLOOR}_gt.json  -> pose RMSE must be ~0 ==="

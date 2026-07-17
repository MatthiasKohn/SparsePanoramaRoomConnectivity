#!/bin/bash
# TRACK F (Leonardo, GPU) — PaGeR per-room geometry (metric depth + normals) over ZInD, then map
# into the pipeline layout so it is a DAP drop-in (--depth_sub pager_depth/depth_meters).
#
# PREREQ (login node, has internet): copy scripts/pager/ZInD_dataloader.py into
#   $HOME/projects/PaGeR/dataloaders/ and add its import + "ZInD" to inference.py's --dataset choices;
#   then WARM-CACHE the weights offline: run PaGeR once with internet ON (any dataset / a couple imgs)
#   so prs-eth/PaGeR + the Depth-Anything-3 backbone land in $HF_HOME.  PaGeR needs ~12 GB VRAM.
#
#   sbatch scripts/trackF_pager_leonardo.slurm            # all homes with panos
#   HOMES_GLOB='0053 0149' sbatch ...                     # (optional) restrict via a filtered root

#SBATCH --job-name=trackF_pager
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/projects/SparsePanoramaRoomConnectivity/scripts/env_leonardo.sh"
cd "$PROJECT_ROOT"

PAGER_DIR="${PAGER_DIR:-$HOME/projects/PaGeR}"
PAGER_PY="${PAGER_PY:-$WORK/envs/fmodels/bin/python}"
PAGER_OUT="${PAGER_OUT:-$WORK/results/pager_zind}"

echo "PaGeR: $PAGER_DIR  py=$PAGER_PY  out=$PAGER_OUT  zind=$ZIND_ROOT"

# 1) run PaGeR inference (indoor scale head) — from the PaGeR repo, its own venv, offline
( cd "$PAGER_DIR" && env -u PYTHONPATH -u LD_LIBRARY_PATH "$PAGER_PY" inference.py \
    --config configs/inference.yaml --checkpoint pager \
    --dataset ZInD --scene_mode indoor \
    --data_path "$ZIND_ROOT" --results_path "$PAGER_OUT" --generate_eval )

# 2) map PaGeR .npz -> <home>/pager_depth/{depth_meters,normals}/<stem>.npy  (pipeline reads these)
python scripts/pager/pager_to_pipeline.py --preds "$PAGER_OUT" --zind_root "$ZIND_ROOT"

echo "Done. Use in the pipeline with:  --depth_sub pager_depth/depth_meters"

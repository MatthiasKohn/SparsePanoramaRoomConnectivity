#!/bin/bash
# TRACK C (Leonardo, GPU) — train the M2 learned camera->door DISTANCE head (exp32).
# Trains on the data_floors substrate (built by Track B); needs NO depth.
#   sbatch scripts/trackC_leonardo.slurm
# Needs the offline caches (DINOv2) + best.pt in runs/hardneg (same prereqs as Track A).

#SBATCH --job-name=trackC_disthead
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/dev/null

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/projects/SparsePanoramaRoomConnectivity/scripts/env_leonardo.sh"
cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out") 2>&1

DATA="${DATA:-$RESULTS_ROOT/../data_floors}"     # where Track B wrote the floor dataset
python -m pipelines.train_distance_head \
    --data "$DATA" --panos_root "$ZIND_ROOT" \
    --ckpt "$RUN_ROOT/hardneg/best.pt" --out "$RUN_ROOT/dist" \
    --val_homes "$RUN_ROOT/hardneg/val_homes.txt" \
    --epochs 20 --bs 128 --unfreeze

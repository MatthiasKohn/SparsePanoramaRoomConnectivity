#!/bin/bash
# TRACK B (Leonardo, CPU) — build the floor-level door dataset (exp30, PaperV2 substrate).
#   sbatch scripts/trackB_leonardo.slurm
# Pure numpy/cv2 (no GPU, no internet). If the cineca-ai module fails on the CPU partition,
# comment out its module load in env_leonardo.sh — the venv alone provides numpy/cv2.

#SBATCH --job-name=trackB_floordata
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod   # CPU job, but the CPU (dcgp) partition needs a
#SBATCH --gres=gpu:1                 # separate budget; boost is where your hours are.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/projects/SparsePanoramaRoomConnectivity/scripts/env_leonardo.sh"
cd "$PROJECT_ROOT"

python -m pipelines.build_floor_dataset --root "$ZIND_ROOT" --with_crops --max 9999 \
    --out "$RESULTS_ROOT/../data_floors"

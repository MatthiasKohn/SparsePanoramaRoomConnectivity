#!/bin/bash
# TRACK B (Leonardo, CPU) — build the floor-level door dataset (exp30, PaperV2 substrate).
#   sbatch scripts/trackB_leonardo.slurm
# Pure numpy/cv2 (no GPU, no internet). If the cineca-ai module fails on the CPU partition,
# comment out its module load in env_leonardo.sh — the venv alone provides numpy/cv2.

#SBATCH --job-name=trackB_floordata
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/projects/SparsePanoramaRoomConnectivity/scripts/env_leonardo.sh"
cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out") 2>&1

python experiments/exp30_build_floor_dataset.py --root "$ZIND_ROOT" --with_crops --max 9999 --out data_floors

#!/bin/bash
# Submit this on Leonardo/CINECA:
#   sbatch scripts/train_leonardo.slurm
#
# Cluster-specific paths/modules live in scripts/env_leonardo.sh.

#SBATCH --job-name=door_train
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "$HOME/projects/SparsePanoramaRoomConnectivity/scripts/env_leonardo.sh"
cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out") 2>&1

RUN_NAME="${RUN_NAME:-unfrozen_nohn}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

mkdir -p "$RUN_DIR"

python -m pipelines.train_embedding \
    --data data_doorpairs \
    --out "$RUN_DIR" \
    --epochs 40 \
    --bs 128 \
    --lr 3e-4 \
    --eval_every 2 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --unfreeze

python -m pipelines.heldout_eval \
    --root "$ZIND_ROOT" \
    --only "$RUN_DIR/val_homes.txt" \
    --ckpt "$RUN_DIR/best.pt" \
    --scoring assign \
    --tag "$RUN_NAME" \
    --max 200

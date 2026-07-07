#!/bin/bash
# Submit this on bwUniCluster:
#   sbatch scripts/train_bwunicluster.slurm
#
# Cluster-specific paths/modules live in scripts/env_bwunicluster.sh.

#SBATCH --job-name=door_train
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/env_bwunicluster.sh
exec > >(tee -a "$LOG_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out") 2>&1

RUN_NAME="${RUN_NAME:-unfrozen_nohn}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

mkdir -p "$RUN_DIR"

python experiments/exp10_train_contrastive.py \
    --data data_doorpairs \
    --out "$RUN_DIR" \
    --epochs 40 \
    --bs 128 \
    --lr 3e-4 \
    --eval_every 2 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --unfreeze

python experiments/exp23_heldout_eval.py \
    --root "$ZIND_ROOT" \
    --only "$RUN_DIR/val_homes.txt" \
    --ckpt "$RUN_DIR/best.pt" \
    --scoring assign \
    --tag "$RUN_NAME" \
    --max 200

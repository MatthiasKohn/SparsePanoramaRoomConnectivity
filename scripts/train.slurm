#!/bin/bash
# Backward-compatible bwUniCluster entrypoint.
# Prefer the explicit cluster scripts:
#   sbatch scripts/train_bwunicluster.slurm
#   sbatch scripts/train_leonardo.slurm

#SBATCH --job-name=door_train
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/env_bwunicluster.sh

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

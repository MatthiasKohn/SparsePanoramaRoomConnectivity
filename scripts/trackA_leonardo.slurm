#!/bin/bash
# TRACK A (Leonardo, GPU) — GT-free connectivity baselines (exp28).
#   sbatch scripts/trackA_leonardo.slurm
#
# PREREQ (Leonardo compute nodes have NO internet): cache the model weights ONCE from a
# LOGIN node, then submit:
#   source scripts/env_leonardo.sh
#   HF_HUB_OFFLINE=0 python -m pipelines.connectivity \
#       --home "$ZIND_ROOT/0001" --ckpt "$RUN_ROOT/hardneg/best.pt" --doors detected \
#       --device cpu --max 1            # downloads DINOv2 + SegFormer into $TORCH_HOME/$HF_HOME
# Also make sure best.pt + val_homes.txt exist under $RUN_ROOT/hardneg (scp from bwUniCluster).

#SBATCH --job-name=trackA_gtfree
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/projects/SparsePanoramaRoomConnectivity/scripts/env_leonardo.sh"
cd "$PROJECT_ROOT"

CKPT="${CKPT:-$RUN_ROOT/hardneg/best.pt}"
VAL="${VAL:-$RUN_ROOT/hardneg/val_homes.txt}"

# detector-driven (the headline GT-free number) + oracle-door control (paired table)
python -m pipelines.connectivity --root "$ZIND_ROOT" --only "$VAL" \
    --ckpt "$CKPT" --doors detected --scoring assign --max 200 --tag hn_det
python -m pipelines.connectivity --root "$ZIND_ROOT" --only "$VAL" \
    --ckpt "$CKPT" --doors gt --scoring assign --max 200 --tag hn_gt

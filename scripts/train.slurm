#!/bin/bash
#SBATCH --job-name=door_train
#SBATCH --partition=gpu_a100_il             # <-- bwUniCluster GPU partition (adjust: gpu_4, gpu_8, gpu_4_a100 ...)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
# Train the contrastive cross-view door embedding.
# Re-submittable: --resume continues from runs/full/last.pt after a timeout.

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs
# Ensure conda is available in non-interactive SLURM shells on bwUniCluster.
module load devel/miniforge
# module load devel/cuda/12.4         # <-- load CUDA if your env needs it
source scripts/activate_env.sh

python experiments/exp10_train_contrastive.py \
    --data data_doorpairs \
    --out  runs/full \
    --epochs 60 --bs 128 --lr 3e-4 --wd 1e-4 \
    --eval_every 2 --workers "$SLURM_CPUS_PER_TASK" \
    --val_frac 0.15 --seed 0 \
    --resume
    # --unfreeze        # uncomment to fine-tune the DINOv2 backbone (needs more GPU mem)

# After training, evaluate connectivity on the HELD-OUT homes the run reserved:
#   python experiments/exp12_connectivity_graph.py \
#       --root <ZIND_ROOT> --only runs/full/val_homes.txt --ckpt runs/full/best.pt

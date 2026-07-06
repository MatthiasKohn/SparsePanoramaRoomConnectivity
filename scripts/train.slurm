#!/bin/bash
#SBATCH --job-name=door_train
#SBATCH --partition=gpu_a100_il
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
#source scripts/activate_env.sh
conda activate roomconn


# frozen + hard-neg (isolates hard-neg from fine-tuning; also rules out the overfit worry)
python experiments/exp10_train_contrastive.py --data data_doorpairs --out runs/hardneg_frozen \
    --epochs 40 --bs 128 --lr 3e-4 --eval_every 2 --workers 8 --hard_neg --homes_per_batch 8
python experiments/exp23_heldout_eval.py --root /home/ul/ul_student/ul_fnm03/data/zind/full_dataset \
    --only runs/hardneg_frozen/val_homes.txt --ckpt runs/hardneg_frozen/best.pt --scoring assign --tag frozen_hn --max 200

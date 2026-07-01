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
source scripts/activate_env.sh

#python experiments/exp10_train_contrastive.py \
#    --data data_doorpairs \
#    --out  runs/full_ft \
#    --epochs 60 --bs 128 --lr 3e-4 --wd 1e-4 \
#    --eval_every 2 --workers "$SLURM_CPUS_PER_TASK" \
#    --val_frac 0.15 --seed 0 \
#    --unfreeze        # uncomment to fine-tune the DINOv2 backbone (needs more GPU mem)


python experiments/exp10_train_contrastive.py --data data_doorpairs --out runs/hardneg \
    --epochs 60 --bs 128 --lr 3e-4 --eval_every 2 --workers 8 --unfreeze --hard_neg --homes_per_batch 8

#python experiments/exp23_heldout_eval.py \
#    --root /home/ul/ul_student/ul_fnm03/data/zind/full_dataset \
#    --only runs/full/val_homes.txt \
#    --ckpt runs/full/best.pt \
#    --mutual \
#    --max 200

#for m in max mutual assign; do
#  python experiments/exp23_heldout_eval.py --root /home/ul/ul_student/ul_fnm03/data/zind/full_dataset \
#      --only runs/full/val_homes.txt --ckpt best.pt --scoring $m --tag $m --max 200
#done
#python experiments/exp25_diagnose_scaling.py --csv results/heldout/heldout_ap_assign.csv


python experiments/exp23_heldout_eval.py --root /home/ul/ul_student/ul_fnm03/data/zind/full_dataset \
 --only runs/hardneg/val_homes.txt --ckpt runs/hardneg/best.pt --scoring assign --tag hardneg --max 200
python experiments/exp25_diagnose_scaling.py --csv results/heldout/heldout_ap_hardneg.csv




# After training, evaluate connectivity on the HELD-OUT homes the run reserved:
#   python experiments/exp12_connectivity_graph.py \
#       --root <ZIND_ROOT> --only runs/full/val_homes.txt --ckpt runs/full/best.pt

#!/bin/bash
#SBATCH --job-name=door_data
#SBATCH --partition=cpu            # <-- bwUniCluster CPU partition (adjust)
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out


# Build the cross-view door-pair dataset from ZInD (CPU only, no GPU needed).
# Usage:  sbatch scripts/build_dataset.slurm /path/to/zind/full_dataset
set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# Ensure conda is available in non-interactive SLURM shells on bwUniCluster.
module load devel/miniforge
source scripts/activate_env.sh

ZIND_ROOT="../data/zind/full_dataset"
python -m pipelines.build_door_dataset \
    --zind_root "$ZIND_ROOT" \
    --out data_doorpairs \
    --fov 70
# tighter crops experiment: add  --fov 50  and use a different --out

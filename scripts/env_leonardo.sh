#!/bin/bash
# Cluster environment for Leonardo/CINECA.
# Source this from SLURM scripts submitted on Leonardo.

set -eo pipefail

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0

source /leonardo_work/EUHPC_D35_121/envs/roomconn/bin/activate

export ZIND_ROOT="${ZIND_ROOT:-/leonardo_work/EUHPC_D35_121/datasets/zind/full_dataset}"
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/SparsePanoramaRoomConnectivity}"
export RUN_ROOT="${RUN_ROOT:-/leonardo_work/EUHPC_D35_121/results/SparsePanoramaRoomConnectivity/runs}"
export LOG_ROOT="${LOG_ROOT:-/leonardo_work/EUHPC_D35_121/logs}"
export RESULTS_ROOT="${RESULTS_ROOT:-/leonardo_work/EUHPC_D35_121/results/SparsePanoramaRoomConnectivity/results}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/leonardo_work/EUHPC_D35_121/checkpoints/SparsePanoramaRoomConnectivity}"

# Leonardo compute nodes are OFFLINE: torch.hub (DINOv2) + HF (SegFormer) weights must
# already live in these caches. Populate them once from a LOGIN node (has internet) — see
# scripts/trackA_leonardo.slurm header.
export TORCH_HOME="${TORCH_HOME:-/leonardo_work/EUHPC_D35_121/cache/torch}"
export HF_HOME="${HF_HOME:-/leonardo_work/EUHPC_D35_121/cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"          # unset on the login node when caching
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$RESULTS_ROOT" "$CHECKPOINT_ROOT" "$TORCH_HOME" "$HF_HOME"

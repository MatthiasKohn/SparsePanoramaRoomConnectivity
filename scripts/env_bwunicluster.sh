#!/bin/bash
# Cluster environment for bwUniCluster.
# Source this from SLURM scripts submitted on bwUniCluster.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

module load devel/miniforge
conda activate roomconn

export ZIND_ROOT="${ZIND_ROOT:-/home/ul/ul_student/ul_fnm03/data/zind/full_dataset}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
export LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$RESULTS_ROOT" "$CHECKPOINT_ROOT"

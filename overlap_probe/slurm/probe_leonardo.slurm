#!/bin/bash
#SBATCH --job-name=overlap_probe
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/projects/SparsePanoramaRoomConnectivity/scripts/env_leonardo.sh"

cd "$PROJECT_ROOT"

exec > >(tee -a "$LOG_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out") 2>&1

# Repositories in $HOME/projects
export ARGUS_DIR="$HOME/projects/RealSee3D"
export PANOVGGT_DIR="$HOME/projects/PanoVGGT"
export VGGT_DIR="$HOME/projects/vggt"

# Separate virtual environments in $WORK/envs
export ARGUS_PY="$WORK/envs/RealSee3D/bin/python"
export PANOVGGT_PY="$WORK/envs/PanoVGGT/bin/python"
export VGGT_PY="$WORK/envs/vggt/bin/python"

# Checkpoints in $WORK/checkpoints
export ARGUS_WEIGHTS="$WORK/checkpoints/RealSee3D"
export PANOVGGT_WEIGHTS="$WORK/checkpoints/PanoVGGT"
export VGGT_WEIGHTS="$WORK/checkpoints/vggt"

# Shared offline caches
export HF_HOME="$WORK/cache/huggingface"
export TORCH_HOME="$WORK/cache/torch"
export XDG_CACHE_HOME="$WORK/cache"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ONLY="${ONLY:-scripts/depth_homes.txt}"
MODELS_TO_RUN="${MODELS_TO_RUN:-argus,panovggt}"

echo "PROJECT_ROOT:      $PROJECT_ROOT"
echo "ZIND_ROOT:         $ZIND_ROOT"
echo "RESULTS_ROOT:      $RESULTS_ROOT"
echo "ARGUS_DIR:         $ARGUS_DIR"
echo "PANOVGGT_DIR:      $PANOVGGT_DIR"
echo "MODELS_TO_RUN:     $MODELS_TO_RUN"
echo "ONLY:              $ONLY"

# Fail early if required paths are missing
for path in \
    "$ARGUS_DIR" \
    "$PANOVGGT_DIR" \
    "$ARGUS_PY" \
    "$PANOVGGT_PY" \
    "$ARGUS_WEIGHTS" \
    "$PANOVGGT_WEIGHTS" \
    "$ONLY"
do
    if [ ! -e "$path" ]; then
        echo "ERROR: Missing path: $path"
        exit 1
    fi
done

python overlap_probe/run_probe.py \
    --root "$ZIND_ROOT" \
    --only "$ONLY" \
    --models "$MODELS_TO_RUN" \
    --out "$RESULTS_ROOT/overlap_probe"
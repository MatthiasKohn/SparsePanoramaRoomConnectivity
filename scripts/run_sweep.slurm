#!/bin/bash
# =============================================================================
# Pose-sensitivity sweep in ONE job: builds the floor many times with GT poses
# perturbed by increasing yaw+translation noise, appends one results.csv row each.
# Metrics-only (no --visuals) so each build is fast; all run on a single GPU node.
#   sbatch scripts/run_sweep.slurm
# Afterwards: the "headline curve" = plot psnr/lpips vs pose_rmse_m from results.csv.
# =============================================================================
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=floor_sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
HOME_ID=0000
FLOOR=floor_01
DEPTH_MODEL=gt_layout                   # keep depth fixed at the oracle while poses vary
DEG_LIST=(0 5 10 15 20 25)              # yaw-noise std in degrees (0 = the oracle point)
SEEDS=(0 1)                             # a couple of seeds per level (noise is random)
M_PER_DEG=0.02                          # translation std = deg * this (deg 25 -> 0.5 m)
FMODELS="/leonardo_work/EUHPC_D35_121/envs/fmodels"
# =============================================================================

set -eo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/.."
source scripts/env_leonardo.sh
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/leonardo_work/EUHPC_D35_121/cache/torch_ext}"
mkdir -p logs "$TORCH_EXTENSIONS_DIR"
PY="$FMODELS/bin/python"
GCCLIB="$(dirname "$(g++ -print-file-name=libstdc++.so.6 2>/dev/null)")"; [[ "$GCCLIB" == /* ]] || GCCLIB=""
RUNPY() { env LD_LIBRARY_PATH="$GCCLIB" TORCH_CUDA_ARCH_LIST="8.0" PYTHONPATH="$PROJECT_ROOT" "$PY" "$@"; }
RUNPY -c "import torch,gsplat;print('env: torch',torch.__version__,'cuda',torch.cuda.is_available())"

for DEG in "${DEG_LIST[@]}"; do
  M=$(python3 -c "print(round($DEG*$M_PER_DEG,3))")
  MODEL=$( [ "$DEG" = 0 ] && echo gt || echo noise )   # deg 0 -> exact GT (the anchor point)
  for S in "${SEEDS[@]}"; do
    echo; echo "########## noise_deg=$DEG noise_m=$M seed=$S (model=$MODEL depth=$DEPTH_MODEL) ##########"
    RUNPY -m pipelines.floor \
        --home "$ZIND_ROOT/$HOME_ID" --floor "$FLOOR" \
        --pose_model "$MODEL" --noise_deg "$DEG" --noise_m "$M" --seed "$S" \
        --depth_model "$DEPTH_MODEL" \
      || echo "   !! deg=$DEG seed=$S failed — continuing"
    [ "$DEG" = 0 ] && break   # GT is deterministic; one seed is enough
  done
done
echo; echo "=== sweep done. rows appended to $RESULTS_ROOT/floor/results.csv ==="

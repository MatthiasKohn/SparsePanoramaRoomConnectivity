#!/bin/bash
# bwUniCluster version of run_floor.slurm (Leonardo is down ~2 weeks).
# Same CONFIG interface; uses the roomconn conda env via scripts/env_bwunicluster.sh.
#   sbatch scripts/run_floor_bwuni.slurm
# Each (home,floor) appends one row to $RESULTS_ROOT/floor/results.csv.
#SBATCH --job-name=floor
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
HOMES=(0000)
FLOORS=(floor_01)
POSE_MODEL=gt                          # gt | noise | drift | salve | badgr | covispose
DEPTH_MODEL=gt_layout                  # gt_layout | fused | pager | dap
COMPLETION=cv2                         # none | cv2 | sd
NOISE="--noise_deg 0 --noise_m 0"      # only used when POSE_MODEL=noise/drift
EXTRA=""                               # add: --visuals (ply+walkthrough), --optimize, --pose_file <json>
# =============================================================================

cd "$SLURM_SUBMIT_DIR"
source scripts/env_bwunicluster.sh      # module load miniforge + conda activate roomconn + ZIND_ROOT/RESULTS_ROOT
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"   # A100
python -c "import torch,gsplat;print('env: torch',torch.__version__,'cuda',torch.cuda.is_available())"

for H in "${HOMES[@]}"; do
  for F in "${FLOORS[@]}"; do
    echo; echo "########## $H / $F  (pose=$POSE_MODEL depth=$DEPTH_MODEL completion=$COMPLETION) ##########"
    python -m pipelines.floor \
        --home "$ZIND_ROOT/$H" --floor "$F" \
        --pose_model "$POSE_MODEL" --depth_model "$DEPTH_MODEL" --completion "$COMPLETION" $NOISE $EXTRA \
      || echo "   !! $H/$F failed (see trace) -- continuing"
  done
done
echo; echo "=== done. results.csv row(s) -> $RESULTS_ROOT/floor/results.csv ==="

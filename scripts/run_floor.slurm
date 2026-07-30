#!/bin/bash
# =============================================================================
# Substitution-study runner: pipelines/floor.py with swappable blocks.
# Edit the CONFIG, `sbatch scripts/run_floor.slurm`. Each (home,floor) appends one row to
# $RESULTS_ROOT/floor/results.csv. Add --visuals in EXTRA to also get the ply + walkthrough.
# One-time for --completion sd:  bash scripts/setup_sd_inpaint.sh
# =============================================================================
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=floor
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
HOMES=(0000)
FLOORS=(floor_01)
POSE_MODEL=gt                          # gt | noise | salve | badgr | covispose
DEPTH_MODEL=gt_layout                      # gt_layout | fused | pager | dap
COMPLETION=cv2                         # none | cv2 | sd
NOISE="--noise_deg 0 --noise_m 0"      # only used when POSE_MODEL=noise
EXTRA="" #--visuals --optimize"           # add/remove: --visuals (ply+walkthrough), --optimize (sharpen)
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

for H in "${HOMES[@]}"; do
  for F in "${FLOORS[@]}"; do
    echo; echo "########## $H / $F  (pose=$POSE_MODEL depth=$DEPTH_MODEL completion=$COMPLETION) ##########"
    RUNPY -m pipelines.floor \
        --home "$ZIND_ROOT/$H" --floor "$F" \
        --pose_model "$POSE_MODEL" --depth_model "$DEPTH_MODEL" --completion "$COMPLETION" $NOISE $EXTRA \
      || echo "   !! $H/$F failed (see trace) — continuing"
  done
done
echo; echo "=== done. results.csv row(s) appended to $RESULTS_ROOT/floor/results.csv ==="

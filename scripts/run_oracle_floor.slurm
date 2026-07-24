#!/bin/bash
# =============================================================================
# GT oracle floor (Paper-2 stage 2a) — GPU / gsplat. One-time: bash scripts/setup_gsplat.sh.
#   sbatch scripts/run_oracle_floor.slurm
# Edit HOMES/FLOORS below. No login-node steps. Outputs -> $RESULTS_ROOT/oracle_floor/<tag>/.
# =============================================================================
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=oracle_floor
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
HOMES=(0000)                          # ZInD homes to process
FLOORS=(floor_01)                     # floors per home (must have >=2-pano connected rooms)
GS_H=1024                             # pano/depth height for GS init
STRIDE=3                              # GS init stride (smaller = denser)
SIZE=512                              # tile render size
YAWS=4                                # held-out tiles per view
WALK="--walkthrough"                  # "" to skip the walkthrough frames
FMODELS="/leonardo_work/EUHPC_D35_121/envs/fmodels"
EXTRA=""
# =============================================================================

set -eo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/.."
source scripts/env_leonardo.sh
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/leonardo_work/EUHPC_D35_121/cache/torch_ext}"
mkdir -p logs "$TORCH_EXTENSIONS_DIR"

# fmodels python directly + PYTHONPATH scrubbed to project root (cineca torch shadow);
# LD_LIBRARY_PATH = only gcc-12 libstdc++ (GLIBCXX for the gsplat kernel), no CUDA shadow.
PY="$FMODELS/bin/python"
GCCLIB="$(dirname "$(g++ -print-file-name=libstdc++.so.6 2>/dev/null)")"
[[ "$GCCLIB" == /* ]] || GCCLIB=""
RUNPY() { env LD_LIBRARY_PATH="$GCCLIB" TORCH_CUDA_ARCH_LIST="8.0" PYTHONPATH="$PROJECT_ROOT" "$PY" "$@"; }
echo "env:"; RUNPY -c "import torch,gsplat;print(' torch',torch.__version__,'gsplat',gsplat.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for H in "${HOMES[@]}"; do
  for F in "${FLOORS[@]}"; do
    TAG="oracle_${H}_${F}"
    echo; echo "########## $H / $F ##########"
    RUNPY -m pipelines.oracle_floor \
        --home "$ZIND_ROOT/$H" --floor "$F" --tag "$TAG" \
        --gs_h "$GS_H" --stride "$STRIDE" --size "$SIZE" --yaws "$YAWS" $WALK $EXTRA \
      || echo "   !! $H/$F failed (see trace) — continuing"
  done
done

echo; echo "=== done. Per floor: metrics.csv (PSNR/SSIM/LPIPS + coverage), heldout_*.png, floor.ply,"
echo "    walkthrough/. This is the UPPER BOUND; substitutions later are measured against it. ==="
echo "First-run QA: check the auto-convention PSNR line (>~22 dB) and one heldout_*.png before trusting numbers."

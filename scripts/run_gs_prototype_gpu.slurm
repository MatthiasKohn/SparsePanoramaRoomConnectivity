#!/bin/bash
# =============================================================================
# Step-0 3D prototype — GPU / gsplat surfel variant. Real alpha hole-map (not the
# CPU density proxy). Requires ONE-TIME `bash scripts/setup_gsplat.sh` first.
#
#   sbatch scripts/run_gs_prototype_gpu.slurm
#
# DEBUG-FIRST: it runs ONE home, ONE view by default so you can eyeball the
# convention (auto-PSNR) + gs_novel.png before spending a sweep. Widen HOMES/FRACS
# only once the first render looks right.
# =============================================================================
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=gs_proto_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
HOMES=(0053)                         # start with ONE home to validate convention
FLOOR="floor_01"
DEPTH_SUB="pager_depth/depth_meters"
NOVEL_FRACS=(0.5)                    # start with ONE view; widen to (door 0.25 0.5 0.75) after
STRIDE=4
ALPHA_THR=0.5                        # alpha below this = hole
GS_FOV=90                            # perspective FOV (deg)
GS_SIZE=512                          # render size (px)
FMODELS="/leonardo_work/EUHPC_D35_121/envs/fmodels"   # torch 2.5.0+cu124 venv (has gsplat)
EXTRA=""
# =============================================================================

set -eo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/.."
source scripts/env_leonardo.sh              # exports PROJECT_ROOT, ZIND_ROOT, RESULTS_ROOT (+ loads cineca-ai)
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/leonardo_work/EUHPC_D35_121/cache/torch_ext}"
mkdir -p logs "$TORCH_EXTENSIONS_DIR"

# CRITICAL: cineca-ai puts its OWN torch on PYTHONPATH, which shadows the fmodels venv (that
# was the libcudnn.so.8 crash — it loaded cineca's torch, not ours). So we call the fmodels
# python binary DIRECTLY with PYTHONPATH = ONLY the project root (no cineca entries), and strip
# LD_LIBRARY_PATH to avoid the nvJitLink shadow. sparsepano/pipelines import from PROJECT_ROOT;
# torch/gsplat/numpy/cv2 come from the venv's own site-packages.
PY="$FMODELS/bin/python"
RUNPY() { env -u LD_LIBRARY_PATH PYTHONPATH="$PROJECT_ROOT" "$PY" "$@"; }
echo "torch/gsplat from:"; RUNPY -c "import torch, gsplat; print(' torch', torch.__version__, '| gsplat', gsplat.__version__, '| cuda', torch.cuda.is_available())"

echo "=== gs_prototype GPU/gsplat run $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "homes=${HOMES[*]}  fracs=${NOVEL_FRACS[*]}  depth=$DEPTH_SUB  size=$GS_SIZE fov=$GS_FOV"

DEPTHNAME=$(basename "$DEPTH_SUB")
for H in "${HOMES[@]}"; do
    echo; echo "########## home $H ##########"
    for F in "${NOVEL_FRACS[@]}"; do
        if [ "$F" = "door" ]; then FRAC_ARG=""; FTAG="door"; else FRAC_ARG="--novel_frac $F"; FTAG="f${F}"; fi
        TAG="gsplat_${H}_${FLOOR}_${DEPTHNAME}_${FTAG}"
        echo "---- novel=$F ----"
        RUNPY -m pipelines.gs_room_prototype \
            --home "$ZIND_ROOT/$H" --floor "$FLOOR" --depth_sub "$DEPTH_SUB" \
            --rasterizer gsplat --alpha_thr "$ALPHA_THR" --gs_fov "$GS_FOV" --gs_size "$GS_SIZE" \
            --stride "$STRIDE" --tag "$TAG" $FRAC_ARG $EXTRA \
          || echo "   !! home $H frac $F failed (see trace above) — continuing"
    done
done

echo; echo "=== done. Check per home: auto-convention PSNR (>18 dB = OK) and gs_novel_holes.png ==="
echo "input-view holes ~0 (control) and NOVEL disocclusion = the real number."

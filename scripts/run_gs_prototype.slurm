#!/bin/bash
# =============================================================================
# Step-0 3D prototype — ONE script you edit + `sbatch` on Leonardo.
# No login-node steps: the pipeline is pure numpy/cv2 (no torch/gsplat/CUDA),
# so everything runs inside this job. Just tweak the CONFIG block and submit:
#
#     sbatch scripts/run_gs_prototype.slurm
#
# Logs -> logs/<jobname>_<jobid>.out ; renders/ply -> $RESULTS_ROOT/gs_prototype/<tag>/
# =============================================================================
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=gs_proto
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# ============================ CONFIG (edit me) ================================
HOMES=(0053 0070 0032 0023)          # homes to process (must have depth generated)
FLOOR="floor_01"
DEPTH_SUB="pager_depth/depth_meters" # or dap_depth/depth_meters to A/B the depth source
NOVEL_FRACS=(door 0.25 0.5 0.75)     # novel-cam position: 'door' = at doorway (best vantage,
                                     #   lower bound); 0..1 = fraction along camA->camB. The
                                     #   sweep gives the disocclusion CURVE as you enter room B.
STRIDE=4                             # pano pixel stride for GS init (↓ = denser, slower)
RENDER_H=512                         # render pano height (width = 2×)
GRID=8                               # coarse coverage cell size (px)
TAU_FRAC=0.25                        # cell well-observed if ≥ tau_frac × median-cell points
EXTRA=""                             # any extra flags passed through to the pipeline
# =============================================================================

set -eo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/.."
source scripts/env_leonardo.sh       # sets PROJECT_ROOT, ZIND_ROOT, RESULTS_ROOT, PYTHONPATH
mkdir -p logs

echo "=== gs_prototype run $(date) ==="
echo "homes=${HOMES[*]}  floor=$FLOOR  depth=$DEPTH_SUB  stride=$STRIDE  render_h=$RENDER_H"
echo "ZIND_ROOT=$ZIND_ROOT  RESULTS_ROOT=$RESULTS_ROOT"

DEPTHNAME=$(basename "$DEPTH_SUB")
for H in "${HOMES[@]}"; do
    echo; echo "########## home $H ##########"
    for F in "${NOVEL_FRACS[@]}"; do
        if [ "$F" = "door" ]; then FRAC_ARG=""; FTAG="door"; else FRAC_ARG="--novel_frac $F"; FTAG="f${F}"; fi
        TAG="zind_${H}_${FLOOR}_${DEPTHNAME}_${FTAG}"
        echo "---- novel=$F ----"
        python -m pipelines.gs_room_prototype \
            --home "$ZIND_ROOT/$H" --floor "$FLOOR" \
            --depth_sub "$DEPTH_SUB" \
            --stride "$STRIDE" --render_h "$RENDER_H" --grid "$GRID" --tau_frac "$TAU_FRAC" \
            --tag "$TAG" $FRAC_ARG $EXTRA \
          || echo "   !! home $H frac $F failed (see trace above) — continuing"
    done
done

echo; echo "=== done. results in $RESULTS_ROOT/gs_prototype/ ==="
echo "Per home: check 'convention check: OK'. Read DISOCCLUSION across novel fracs = the curve."
echo "To eyeball geometry locally, copy a few files, e.g.:"
echo "  scp -r LEONARDO:$RESULTS_ROOT/gs_prototype/zind_0070_${FLOOR}_${DEPTHNAME}_door ."

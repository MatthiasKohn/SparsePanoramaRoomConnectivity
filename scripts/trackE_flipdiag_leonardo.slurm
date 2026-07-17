#!/bin/bash
# TRACK E (Leonardo, GPU) — diagnose exp29's below-chance flip prior (0.45).
#   sbatch scripts/trackE_flipdiag_leonardo.slurm
#
# No new depth needed for the 20-home diagnostic: it reuses the dap_depth trackD produced.
# It ALSO regenerates depth for home 0025 (the original 7/7 anecdote, NOT in depth_homes.txt)
# and re-runs it through the exp29 harness, to check the anecdote still reproduces there.
#
# Reads: $RUN_ROOT/hardneg/best.pt , $ZIND_ROOT/<home>/dap_depth/depth_meters/*.npy
# Writes: $RESULTS_ROOT/floorbench/flipdiag_hn.csv  (+ stdout verdict)

#SBATCH --job-name=trackE_flipdiag
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/projects/SparsePanoramaRoomConnectivity/scripts/env_leonardo.sh"
cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out") 2>&1

HOMES="${HOMES:-scripts/depth_homes.txt}"
CKPT="${CKPT:-$RUN_ROOT/hardneg/best.pt}"

echo "############ 1) FLIP-PRIOR DIAGNOSTIC over the 20 benchmark homes ############"
python -m pipelines.flip_diagnose \
    --root "$ZIND_ROOT" --homes "$HOMES" \
    --ckpt "$CKPT" --device cuda --tag hn

echo
echo "############ 2) REPRODUCE the 0025 anecdote through the exp29 harness ############"
# 0025 is the original 7/7 flip result but is NOT in depth_homes.txt, so generate its depth.
if [ ! -d "$ZIND_ROOT/0025/dap_depth/depth_meters" ]; then
    echo "=== DAP depth: 0025 ==="
    python scripts/generate_depth.py \
        --input_dir "$ZIND_ROOT/0025/panos" \
        --output_dir "$ZIND_ROOT/0025/dap_depth" --pattern "*.jpg"
fi
python -m pipelines.pose_layout \
    --root "$ZIND_ROOT" --homes 0025 \
    --ckpt "$CKPT" --device cuda --tag repro0025
echo "If 0025 flip_emb is high here but the 20-home mean is ~0.45, the prior works on"
echo "some floors and fails on others -> a reliability/gating story, not a dead mechanism."

#!/bin/bash
# TRACK D (Leonardo, GPU) — DAP depth for the 20 held-out cyclic homes, then exp29 pose/flip.
#   sbatch scripts/trackD_leonardo.slurm
#
# PREREQ 1 — DAP repo + weights (Leonardo compute nodes have NO internet):
#   generate_depth.py wraps  $HOME/projects/DAP/test/run_dap_inference.py  (config.DAP_ROOT =
#   PROJECT_ROOT.parent/"DAP"). It must sit BESIDE the project: $HOME/projects/DAP.
#   From your Windows box (where DAP already runs), copy the whole repo incl. weights:
#     scp -r C:\...\DAP  <user>@login.leonardo.cineca.it:~/projects/DAP
#   Then install its deps into the roomconn env from a LOGIN node (has internet):
#     source scripts/env_leonardo.sh
#     pip install -r ~/projects/DAP/requirements.txt        # timm / DINOv3 backbone etc.
#   DAP's backbone weights must be offline too: run ONE image on the login node so any
#   torch.hub/HF pull is cached into $TORCH_HOME/$HF_HOME (HF_HUB_OFFLINE=0 for that one call):
#     HF_HUB_OFFLINE=0 python scripts/generate_depth.py \
#        --input_dir "$ZIND_ROOT/0025/panos" --output_dir "$ZIND_ROOT/0025/dap_depth" --pattern "*.jpg"
#
# PREREQ 2 — best.pt + val_homes.txt under $RUN_ROOT/hardneg (scp from bwUniCluster, as Track A).
# PREREQ 3 — scripts/depth_homes.txt already lists the 20 homes (built by find_cyclic_homes.py).

#SBATCH --job-name=trackD_depth_exp29
#SBATCH --account=EUHPC_D35_121
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/projects/SparsePanoramaRoomConnectivity/scripts/env_leonardo.sh"
cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG_ROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out") 2>&1

HOMES="${HOMES:-scripts/depth_homes.txt}"
CKPT="${CKPT:-$RUN_ROOT/hardneg/best.pt}"

# 1) DAP depth for each held-out home -> $ZIND_ROOT/<home>/dap_depth/depth_meters/*.npy
while read -r H; do
    [ -z "$H" ] && continue
    echo "=== DAP depth: $H ==="
    python scripts/generate_depth.py \
        --input_dir  "$ZIND_ROOT/$H/panos" \
        --output_dir "$ZIND_ROOT/$H/dap_depth" \
        --pattern "*.jpg"
done < "$HOMES"

# 2) pose/flip benchmark over every now-depth-equipped floor (exp29 reads dap_depth/depth_meters)
python experiments/exp29_floor_benchmark.py \
    --root "$ZIND_ROOT" --homes "$HOMES" \
    --ckpt "$CKPT" --device cuda --tag hn

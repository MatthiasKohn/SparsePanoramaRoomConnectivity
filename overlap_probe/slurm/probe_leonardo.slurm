#!/bin/bash
# overlap_probe (Leonardo, GPU) — do Argus/PanoVGGT/VGGT actually solve our sparse setting?
#   sbatch overlap_probe/slurm/probe_leonardo.slurm
#
# PREREQ (compute nodes are OFFLINE — stage everything from a login node first):
#   1. Model repos + weights checked out under /leonardo_work/EUHPC_D35_121/models/:
#        ARGUS_DIR, PANOVGGT_DIR, VGGT_DIR  (export below to point at them)
#      Wire each adapter's _cmd() in overlap_probe/adapters.py to the repo's real inference
#      call (each must emit out/pred.npz with key 'poses' = (N,4,4) c2w).
#   2. Any torch.hub/HF backbone weights the models pull must already be in $TORCH_HOME/$HF_HOME
#      (run one demo image per model on the login node with HF_HUB_OFFLINE=0 to cache).
#   3. Validate the harness first WITHOUT the models (oracle/noisy) — see the last line.

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

M=/leonardo_work/EUHPC_D35_121/models
export ARGUS_DIR="${ARGUS_DIR:-$M/argus}";       export ARGUS_PY="${ARGUS_PY:-$M/argus/.venv/bin/python}"
export PANOVGGT_DIR="${PANOVGGT_DIR:-$M/panovggt}"; export PANOVGGT_PY="${PANOVGGT_PY:-$M/panovggt/.venv/bin/python}"
export VGGT_DIR="${VGGT_DIR:-$M/vggt}";          export VGGT_PY="${VGGT_PY:-$M/vggt/.venv/bin/python}"

ONLY="${ONLY:-scripts/depth_homes.txt}"     # the 20 held-out cyclic homes (or a bigger list)

python overlap_probe/run_probe.py \
    --root "$ZIND_ROOT" --only "$ONLY" \
    --models argus,panovggt,vggt_tiled \
    --out "$RESULTS_ROOT/overlap_probe"

# Harness sanity (no models needed), if you want it in the same job:
# python overlap_probe/run_probe.py --root "$ZIND_ROOT" --only "$ONLY" --models oracle,noisy \
#     --out "$RESULTS_ROOT/overlap_probe_selftest"

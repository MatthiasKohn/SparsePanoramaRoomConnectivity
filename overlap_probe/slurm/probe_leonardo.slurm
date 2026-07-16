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

# --- Model CODE repos (each with overlap_probe_infer.py copied in) -----------------
# NB: Argus CODE lives in github.com/realsee-developer/Argus — NOT the RealSee3D dataset repo.
export ARGUS_DIR="${ARGUS_DIR:-$HOME/projects/Argus}"
export PANOVGGT_DIR="${PANOVGGT_DIR:-$HOME/projects/PanoVGGT}"

# --- Per-model virtual environments (own torch versions) --------------------------
export ARGUS_PY="${ARGUS_PY:-$WORK/envs/Argus/bin/python}"
export PANOVGGT_PY="${PANOVGGT_PY:-$WORK/envs/PanoVGGT/bin/python}"

# --- Weights / configs the wrappers read (FILES, not dirs) ------------------------
export ARGUS_CKPT="${ARGUS_CKPT:-$WORK/checkpoints/Argus/argus_realsee3d.pt}"
export ARGUS_H="${ARGUS_H:-518}"                       # ERP input height (width=2*H)
export PANOVGGT_CKPT="${PANOVGGT_CKPT:-$WORK/checkpoints/PanoVGGT/model.pt}"
export PANOVGGT_CONFIG="${PANOVGGT_CONFIG:-$PANOVGGT_DIR/training/config/default.yaml}"

# --- Shared offline caches --------------------------------------------------------
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
echo "ARGUS_DIR:         $ARGUS_DIR   (ckpt: $ARGUS_CKPT)"
echo "PANOVGGT_DIR:      $PANOVGGT_DIR (ckpt: $PANOVGGT_CKPT)"
echo "MODELS_TO_RUN:     $MODELS_TO_RUN"
echo "ONLY:              $ONLY"

# Fail early if required paths are missing — only for the models actually requested
REQUIRED=("$ONLY")
case ",$MODELS_TO_RUN," in
  *,argus,*)    REQUIRED+=("$ARGUS_DIR/overlap_probe_infer.py" "$ARGUS_PY" "$ARGUS_CKPT");;
esac
case ",$MODELS_TO_RUN," in
  *,panovggt,*) REQUIRED+=("$PANOVGGT_DIR/overlap_probe_infer.py" "$PANOVGGT_PY" "$PANOVGGT_CKPT" "$PANOVGGT_CONFIG");;
esac
for path in "${REQUIRED[@]}"; do
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
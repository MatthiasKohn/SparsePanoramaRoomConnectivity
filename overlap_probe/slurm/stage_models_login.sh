#!/bin/bash
# Run once on a Leonardo login node with internet access.
set -euo pipefail

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0

PROJECTS="$HOME/projects"
ENVS="$WORK/envs"
CHECKPOINTS="$WORK/checkpoints"

export HF_HOME="$WORK/cache/huggingface"
export TORCH_HOME="$WORK/cache/torch"
export XDG_CACHE_HOME="$WORK/cache"

mkdir -p \
    "$PROJECTS" \
    "$ENVS" \
    "$CHECKPOINTS" \
    "$HF_HOME" \
    "$TORCH_HOME"

clone_and_env() {
    local name="$1"
    local url="$2"
    local repo_dir="$PROJECTS/$name"
    local env_dir="$ENVS/$name"

    if [ ! -d "$repo_dir/.git" ]; then
        git clone "$url" "$repo_dir"
    else
        echo "Repository already exists: $repo_dir"
    fi

    if [ ! -d "$env_dir" ]; then
        python -m venv --system-site-packages "$env_dir"
    fi

    # shellcheck disable=SC1091
    source "$env_dir/bin/activate"

    python -m pip install --upgrade pip setuptools wheel

    if [ -f "$repo_dir/requirements.txt" ]; then
        python -m pip install -r "$repo_dir/requirements.txt"
    else
        echo "No requirements.txt found for $name; check its README."
    fi

    deactivate

    mkdir -p "$CHECKPOINTS/$name"

    echo "== $name =="
    echo "Repository:  $repo_dir"
    echo "Environment: $env_dir"
    echo "Weights:     $CHECKPOINTS/$name"
}

clone_and_env RealSee3D \
    https://github.com/realsee-developer/RealSee3D.git

clone_and_env PanoVGGT \
    https://github.com/YijingGuo-June/PanoVGGT.git

clone_and_env vggt \
    https://github.com/facebookresearch/vggt.git

echo
echo "Repositories installed under: $PROJECTS"
echo "Environments installed under: $ENVS"
echo "Weights should be stored under: $CHECKPOINTS"
echo "Hugging Face cache: $HF_HOME"
echo "Torch cache: $TORCH_HOME"
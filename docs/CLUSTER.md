# Running the connectivity pipeline on bwUniCluster

Goal: train the contrastive cross-view **door embedding** on full ZInD, then report
**held-out room-connectivity Average Precision** (the headline metric).

Pipeline:  `exp09` build dataset → `exp10` train → `exp12` eval on held-out homes.

## 0. Environment (once)
```
module load devel/miniforge
conda activate roomconn
pip install -r requirements.txt
```
PyTorch is installed separately because the correct CUDA build is cluster-specific.
Current verified setup on an A100 node:

```
conda activate roomconn
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Expected/current:

```
2.12.1+cu130
True
NVIDIA A100-PCIE-40GB
```

Then install the remaining project dependencies:

```
pip install -r requirements.txt
```

The SLURM scripts source `scripts/activate_env.sh`, which defaults to:

```
roomconn
```

To use a different env without editing the scripts:

```
export ROOMCONN_CONDA_ENV=/path/to/other/env
```

DINOv2 weights download once via torch.hub on first run. If compute nodes are offline,
run one tiny login/precompute job first to cache them. The scripts set `TORCH_HOME`
and `HF_HOME` to cache locations under `$HOME/.cache` unless you override them.

## 1. Build the door-pair dataset (CPU)
```
sbatch scripts/build_dataset.slurm /path/to/zind/full_dataset
# -> data_doorpairs/  (crops/ + pairs.csv).  exp09 REBUILDS fresh by default (no dup-append).
```

## 2. Train (GPU, resumable)
```
sbatch scripts/train.slurm
```
- Writes to `runs/full/`: `last.pt` (resume), `best.pt` (best val top5), `door_encoder.pt`,
  `split.json`, **`val_homes.txt`** (the held-out homes), and `train_log.csv`.
- Monitor: `tail -f runs/full/train_log.csv` (epoch, loss, val_top1, val_top5).
- **Timeout?** Just `sbatch scripts/train.slurm` again — `--resume` continues from `last.pt`.
- More capacity: add `--unfreeze` (fine-tune the backbone; needs more GPU mem).

## 3. Evaluate connectivity on HELD-OUT homes (the headline number)
```
python -m pipelines.connectivity_graph \
    --root /path/to/zind/full_dataset \
    --only runs/full/val_homes.txt \
    --ckpt runs/full/best.pt
# -> per-home AP + "MEAN AP over N homes".  This is the defensible result.
```
`--only val_homes.txt` guarantees you only score homes the model never trained on.

## Ablations (for the paper)
- **Data scaling** (the key figure — held-out AP vs #homes): `bash scripts/scaling_curve.sh <ZIND_ROOT>`
- **Crop size**: rebuild with `--fov 50` (tighter) to a new `--out`, retrain, compare.
- **Backbone**: `--unfreeze` vs frozen.
- **Probe what the embedding uses**: mask the door region vs the through-door region in
  the eval crops and see which hurts retrieval more (door-object cue vs see-through cue).

## bwUniCluster notes
- Adjust `--partition`, `--time`, `--gres`, and any `module load` lines in the `.slurm`
  files to your allocation. Logs go to `logs/`.
- `runs/`, `data_doorpairs/`, `*.pt` are git-ignored — keep the repo code-only.
- Reproducibility: everything is seeded (`--seed`); the train/val split is saved to
  `runs/full/split.json`.

## Floor graph + embedding flip-prior (exp18)
Once `best.pt` exists, build a globally consistent floor map from door edges:
```
python -m pipelines.floor_graph_real --ckpt best.pt --device cuda
```
- Front-end: best-viewing pano pair per room-pair -> door-anchored measured pose +
  which-side flip candidate (inlier-weighted).
- Flip prior: the trained door encoder (through-door reprojection-consistency).
- Output: `results/floorgraph/floor_real_sampletour.png` (GT vs geometry vs +embedding)
  and per-edge true-side accuracy + room-position error.
- Needs DAP depth for the panos; only the sample tour has it locally. To run on a full
  multi-room home with real cycles, generate depth first (`scripts/generate_depth.py`).

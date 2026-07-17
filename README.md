# SparsePanoramaRoomConnectivity

Dataset-agnostic sparse panorama pipeline for room connectivity, door-anchored
pose/layout, and later 3D reconstruction.

The scientific target is:

```text
panos -> metric geometry/depth -> doors + cross-view embedding
      -> connectivity graph + door-relative pose
      -> global floor layout
      -> later feed-forward 3DGS / diffusion conditioning
```

## Layout

```text
sparsepano/   core library: datasets, geometry, doors, pose, gs, metrics, viz
pipelines/    runnable stage CLIs and the unified runner
benchmarks/   standalone benchmarks, including overlap_probe
configs/      dataset/run config examples
scripts/      cluster env and SLURM scripts
docs/         project notes and migration docs
tests/        smoke and regression tests
legacy/       archived one-off experiments kept for provenance
weights/      local checkpoints; gitignored
```

Generated artifacts such as `results/`, `runs/`, `logs/`, `data_*`, and cache
archives are intentionally gitignored.

## Install

```bash
pip install -e .
```

PyTorch is not pinned in `pyproject.toml`; install the CUDA build appropriate for
your machine/cluster before running GPU stages.

## Run Connectivity

Using explicit paths:

```bash
python -m pipelines.run \
  --dataset zind \
  --root "$ZIND_ROOT" \
  --split heldout \
  --only "$RUN_ROOT/hardneg/val_homes.txt" \
  --stage connectivity \
  --doors gt \
  --ckpt "$RUN_ROOT/hardneg/best.pt" \
  --scoring assign \
  --max 200 \
  --out results/zind_gt_connectivity
```

With Leonardo env/config:

```bash
source scripts/env_leonardo.sh
python -m pipelines.run --config configs/zind_leonardo.yaml \
  --stage connectivity --doors detected \
  --ckpt "$RUN_ROOT/hardneg/best.pt" \
  --out "$RESULTS_ROOT/trackA_detected"
```

Each run writes:

```text
metrics.json
report.md
per-stage CSV/plots under the run directory
```

Regression targets for the validated ZInD held-out split are approximately:

- GT doors, assign scoring: AP `0.913`
- Detected doors, assign scoring: AP `0.842`

## Tests

Fast contract/import checks:

```bash
pytest tests/test_dataset_contract.py
```

Slow connectivity regression, requiring ZInD and weights:

```bash
pytest tests/test_connectivity_regression.py --run-slow \
  --zind-root "$ZIND_ROOT" \
  --heldout "$RUN_ROOT/hardneg/val_homes.txt" \
  --ckpt "$RUN_ROOT/hardneg/best.pt"
```

## Adding A Dataset

Implement `sparsepano.datasets.base.Dataset`, convert native annotations into
`Scene` / `Pano` / `Door`, set capability flags, and register the adapter with
`@register_dataset("name")`. See `sparsepano/datasets/README.md`.

## Cluster Notes

Cluster-specific setup remains in `scripts/env_*.sh` and `scripts/*.slurm`.
Leonardo compute nodes are offline, so model caches must be pre-populated under
the cache roots exported by `scripts/env_leonardo.sh`.


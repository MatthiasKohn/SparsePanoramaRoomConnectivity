# Migration Plan

This repo is being refactored incrementally into a dataset-agnostic sparse panorama
pipeline. The rule for the migration is: preserve validated behaviour first, then
move code behind cleaner interfaces.

## Target Tree

```text
sparsepano/
  datasets/       Dataset abstraction and adapters.
  geometry/       Projection, cubemap, depth, normals, geometry providers.
  doors/          Door detection, embedding, crop/dataset building.
  pose/           Door-relative pose, pose graph, registration, matching.
  gs/             3DGS-related initialisation/optimisation.
  metrics/        Pure metric functions and diagnostics.
  viz/            Visualisations, overlays, point clouds.
pipelines/        Dataset-agnostic CLIs and stage runners.
benchmarks/       Validated standalone benchmarks, including overlap_probe.
configs/          Dataset/run configuration examples.
scripts/          Cluster environment and SLURM scripts.
legacy/           Superseded experiment scripts kept for provenance.
docs/             Project docs and canonical ContextMDs.
tests/            Smoke and regression checks.
results/          Generated reports/plots/CSVs; gitignored.
weights/          Local model weights/checkpoints; gitignored unless explicitly tracked.
```

## Phases

1. [done] Scaffolding: add package layout, editable install metadata, config examples,
   and this manifest.
2. [done] Dataset abstraction: add `Scene`/`Pano`/`Door` dataclasses, registry, and a
   ZInD adapter while leaving old `src.zind` usable.
3. [done] Bridge pipelines: add `python -m pipelines.run` and route connectivity through
   the validated connectivity implementation so known AP numbers remain reproducible.
4. [done] Metrics/reporting: add pure metric modules and report writers, then migrate
   diagnostics out of experiments.
5. [done] Topical migration: move reusable source modules into `sparsepano/*` packages.
6. [done] Archive one-offs: move superseded experiment scripts to `legacy/`
   after reusable helpers are migrated.
7. [done] Regression tests: add slow AP checks and fast selftests for plumbing.
8. [done] Documentation: update README and cluster-facing paths.

## Completed Cleanup

- Deleted top-level `former src/`; implementations now live in `sparsepano/{datasets,geometry,doors,pose,gs,viz}`.
- Deleted top-level `experiments/`; migrated implementations live in `pipelines/`, archived one-offs in
  `legacy/experiments/`.
- Moved `overlap_probe/` to `benchmarks/overlap_probe/`.
- Moved canonical docs and migration prompts to `docs/`.
- Deleted deprecated root/doc stubs: `MeetingSummary.md`, `ProposalDraft.md`, and deprecated `ContextMDs/*`.
- Deleted root `config.py`; use `sparsepano.config`.
- Moved loose checkpoints to `weights/`.

## Current File Mapping

Status values:

- `keep-now`: still used in-place during bridge phase.
- `migrate`: reusable code should move into a topical package.
- `pipeline`: experiment logic should become a dataset-agnostic runner.
- `legacy`: likely one-off after helpers are extracted.
- `generated`: keep on disk if useful, but gitignore / do not treat as source.
- `docs`: move or rewrite under `docs/`.
- `cluster`: keep under `scripts/`, update paths as modules move.

| Current path | Status | Destination / note |
| --- | --- | --- |
| `config.py` | migrate | `sparsepano/config.py`; keep compatibility until imports are moved. |
| `requirements.txt` | keep-now | Keep; mirrored by `pyproject.toml` optional dependencies. |
| `README.md` | docs | Rewrite after CLI stabilises. |
| `CLUSTER.md` | docs | Move/update under `docs/` after SLURM paths settle. |
| `AGENTS.md` | keep-now | Local collaboration instructions. |
| `ContextMDs/` | docs | Move canonical non-deprecated docs to `docs/`; remove deprecated stubs. |
| `MeetingSummary.md` | docs | User-edited; do not touch without explicit approval. |
| `ProposalDraft.md` | docs | User-edited; do not touch without explicit approval. |
| `TODO.md` | docs | User-edited; do not touch without explicit approval. |
| `NEW_CHAT_PROMPT.md` | docs | User-edited; do not touch without explicit approval. |
| `CODEX_REFACTOR_PROMPT.md` | docs | Refactor prompt/provenance; later move to `docs/`. |
| `.gitignore` | keep-now | Extend for generated artifacts. |
| `.gitattributes` | keep-now | Keep. |
| `.venv-gs/`, `.idea/`, `.tools/`, `.agents/`, `.codex/`, `__pycache__/` | generated | Local only / ignored. |
| `data_doorpairs/`, `data_floors/` | generated | Rebuildable datasets; keep local, ignored. |
| `runs/`, `results/`, `logs/` | generated | Generated outputs; final reports go under `results/<run_id>/`. |
| `torchhub.tgz` | generated | Transfer artifact; ignored. |
| `best.pt`, `best_hardneg.pt`, `door_encoder.pt` | generated | Preserve; later move/copy to `weights/` with LFS/ignore policy. |
| `former src/zind.py` | migrate | `sparsepano/datasets/zind.py`; keep compatibility until old imports are gone. |
| `former src/stanford.py` | migrate | Candidate `sparsepano/datasets/stanford.py`; if stale, move to legacy with `TODO(codex)`. |
| `former src/providers.py` | migrate | `sparsepano/geometry/providers.py`; dataset access must go through adapters. |
| `former src/panoproj.py` | migrate | `sparsepano/geometry/panoproj.py`. |
| `former src/geom.py` | migrate | `sparsepano/geometry/geom.py`. |
| `former src/pose.py`, `sparsepano/pose/posegraph.py`, `sparsepano/pose/door_pose.py`, `former src/register.py`, `former src/matching.py` | migrate | `sparsepano/pose/`. |
| `former src/doors.py`, `sparsepano/doors/door_semantic.py`, `former src/door_dataset.py`, `former src/contrastive.py`, `former src/hard_neg.py`, `former src/aperture.py` | migrate | `sparsepano/doors/`. |
| `former src/gs_optim.py`, `sparsepano/gs/gsplat_init.py` | migrate | `sparsepano/gs/`. |
| `tools/viewer.py` | migrate | `sparsepano/viz/` or `tools/` wrapper. |
| `overlap_probe/` | migrate | Move unchanged to `benchmarks/overlap_probe/`; fix external paths only. |
| `experiments/exp09_build_door_dataset.py` | pipeline | `pipelines/build_door_dataset.py`. |
| `experiments/exp10_train_contrastive.py` | pipeline | `pipelines/train_embedding.py`. |
| `experiments/exp12_connectivity_graph.py`, `experiments/exp23_heldout_eval.py` | pipeline | `pipelines/connectivity.py` + `sparsepano/metrics/connectivity.py`. |
| `experiments/exp28_gtfree_connectivity.py` | pipeline | Bridge evaluator for `pipelines.run --stage connectivity`; later fold into metrics/pipeline. |
| `experiments/exp18_floor_graph_real.py`, `experiments/exp27_hybrid_real.py` | pipeline | `pipelines/pose_layout.py`. |
| `experiments/exp29_floor_benchmark.py` | pipeline | `pipelines/pose_layout.py` + `sparsepano/metrics/pose.py`. |
| `experiments/exp30_build_floor_dataset.py` | pipeline | Floor-pair dataset builder. |
| `experiments/exp31_distance_baseline.py`, `experiments/exp32_train_distance_head.py` | migrate | `sparsepano/pose/` and training pipeline. |
| `experiments/exp33_flip_diagnose.py` | migrate | `sparsepano/metrics/pose.py` diagnostics/report. |
| `experiments/exp01_pose_ablation.py`, `exp02_geometric_detection.py`, `exp04_immersight.py`, `exp05_doorway_detect.py`, `exp06_semantic_doors.py`, `exp07_door_match.py`, `exp08_whichside.py`, `exp11_whichside_trained.py`, `exp13_immersight_match.py`, `exp15_match_saliency.py`, `exp16_photometric_pose.py`, `exp17_floor_graph.py`, `exp19_gs_room.py`, `exp20_gs_pair.py`, `exp21_gs_pair_estimated.py`, `exp22_gs_floor.py`, `exp24_colmap_compare.py`, `exp25_diagnose_scaling.py`, `exp26_hybrid_posegraph.py` | legacy | Scan for helpers before moving; then archive under `legacy/experiments/`. |
| `experiments/exp14_export_pointcloud.py` | migrate | `sparsepano/viz/pointcloud.py` CLI. |
| `scripts/env_bwunicluster.sh`, `scripts/env_leonardo.sh` | cluster | Keep. Ensure `PYTHONPATH`/editable install works on clusters. |
| `scripts/train*.slurm`, `scripts/track*.slurm`, `scripts/build_dataset.slurm` | cluster | Keep and update to call `pipelines.run` once bridge is validated. |
| `scripts/generate_depth.py`, `scripts/precache_weights.py` | cluster | Keep wrappers; later route model roots through config. |
| `scripts/find_cyclic_homes.py`, `scripts/scaling_curve.sh`, `scripts/paper_runs.sh`, `scripts/run_colmap_home.sh`, `scripts/run_hybrid_home.sh`, `scripts/colmap_perspective.py`, `scripts/rescale_depth.py` | cluster | Keep now; migrate reusable pieces to pipelines/pose/viz. |
| `scripts/depth_homes.txt`, `scripts/cyclic_floors.csv` | keep-now | Small run inputs; later move to `configs/splits/` if stable. |

## Regression Targets

- ZInD held-out connectivity assign-AP with GT doors: approximately `0.913`.
- ZInD held-out connectivity assign-AP with detected doors: approximately `0.842`.

These remain backed by the existing `exp28`/`exp12` code until the metric layer is
fully ported. Any migration that changes these numbers needs an explicit scientific
reason, not an incidental refactor.


# TODO (open action items)

## Done
- Repo restructured into `sparsepano/` + `pipelines/` + `benchmarks/` + `docs/` + `legacy/` (dataset-
  agnostic; old `src/`/`experiments/` migrated). Connectivity reproduces (`pipelines.run --stage
  connectivity`, AP ≈0.91). Cluster scripts migrated to `python -m pipelines.*`; `env_*.sh` set PYTHONPATH.

## Next
- [ ] **Regenerate clean PanoVGGT overlap_probe result** (argus-only runs overwrote CSV/plots):
      `MODELS_TO_RUN=panovggt sbatch benchmarks/overlap_probe/slurm/probe_leonardo.slurm`.
- [ ] **Adopt PaGeR** as per-room geometry backbone (depth+normals); retire DAP. Add
      `sparsepano/geometry/pager.py` provider + a cluster staging track (see chat plan).
- [ ] **Close the 0.913→0.842 connectivity gap** = detector RECALL: cubemap-face (undistorted) door
      detection + PaGeR geometric opening proposals. Simplest test: recall@15° cubemap vs ERP.
- [ ] Wire the unified `run.py` orchestrator stages (geometry/doors/pose/layout) — currently stubs.
- [ ] Paper 1 remaining: real SfM/COLMAP anchor run (C5), ablation grid + hygiene retrain (G4/G5).
- [ ] Paper 2 prototype: GT-pose 2–3 rooms → per-room GS → off-the-shelf view-diffusion completion of
      one through-door transition (kill criterion first).

# TODO (open action items)

## Done
- Repo restructured (dataset-agnostic package); connectivity reproduces (0.913 GT / 0.842 detected).
- Cluster scripts migrated to `python -m pipelines.*`; logs → `logs/`; single shared `fmodels` env.
- overlap_probe: 2D + 3D point-cloud viz; clean PanoVGGT result (dense 5.7° / sparse 23.9°).
- PaGeR wired + depth/normals generated on 20 held-out homes; qualitatively >> DAP (doorways resolved).

## Next (priority order)
- [ ] **Close the PaGeR decision:** `pipelines.distance_baseline` PaGeR vs DAP (depth→door distance vs GT).
      If PaGeR wins → adopt as backbone + retire DAP. Also retest the *free-space* which-side flip with PaGeR depth.
- [ ] **Connectivity recall lever (0.913→0.842):** cubemap-face (undistorted) door detection + PaGeR
      geometric opening proposals. Simplest test: recall@15° cubemap vs ERP on the held-out homes.
- [ ] **Add Stanford2D3DS** to the dataset abstraction → (a) DIRECT depth metric (AbsRel/RMSE/δ) for
      PaGeR/DAP, (b) an in-distribution overlap_probe run (addresses the ZInD-OOD caveat).
- [ ] Paper 1 loose ends: real SfM/COLMAP anchor (C5); ablation grid + hygiene retrain (G4/G5); wire the
      unified `run.py` orchestrator stages (geometry/doors/pose/layout are stubs).
- [ ] Paper 2 prototype: GT-pose 2–3 rooms → per-room GS → off-the-shelf view-diffusion completion of one
      through-door transition (kill criterion first).

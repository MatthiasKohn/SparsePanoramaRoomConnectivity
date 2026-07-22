# TODO (open action items)

## Done
- Repo restructured (dataset-agnostic package); connectivity reproduces (0.913 GT / 0.842 detected).
- Cluster scripts migrated to `python -m pipelines.*`; logs → `logs/`; single shared `fmodels` env.
- overlap_probe: 2D + 3D point-cloud viz; clean PanoVGGT result (dense 5.7° / sparse 23.9°).
- PaGeR wired + depth/normals generated on 20 held-out homes; qualitatively >> DAP (doorways resolved).

## Next (priority order)
- [x] **PaGeR distance A/B done:** DAP beats PaGeR on camera→door distance (0.69 vs 0.93 m median, 16/20
      homes); fixed-width geometry best on median (0.61) → learned per-door distance head is the lever, not
      a depth swap. PaGeR NOT adopted for distance; kept for normals/geometry roles.
- [ ] **Learned per-door distance head (M2, exp32→pipeline):** beat fixed-width's 0.61 m median WITHOUT its
      wide-opening MAE blowup — the real distance lever (no monocular depth).
- [ ] **Use PaGeR where it helps:** (a) geometric opening proposals for connectivity RECALL, (b) free-space
      which-side flip retest with PaGeR geometry, (c) Paper-2 per-room 3D.
- [ ] **Connectivity recall lever (0.913→0.842):** cubemap-face (undistorted) door detection + PaGeR
      geometric opening proposals. Simplest test: recall@15° cubemap vs ERP on the held-out homes.
- [ ] **Add Stanford2D3DS** to the dataset abstraction → (a) DIRECT depth metric (AbsRel/RMSE/δ) for
      PaGeR/DAP, (b) an in-distribution overlap_probe run (addresses the ZInD-OOD caveat).
- [ ] Paper 1 loose ends: real SfM/COLMAP anchor (C5); ablation grid + hygiene retrain (G4/G5); wire the
      unified `run.py` orchestrator stages (geometry/doors/pose/layout are stubs).
- [ ] Paper 2 prototype: GT-pose 2–3 rooms → per-room GS → off-the-shelf view-diffusion completion of one
      through-door transition (kill criterion first).

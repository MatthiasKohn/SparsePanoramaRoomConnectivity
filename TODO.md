# TODO (open action items)

- [ ] **Repo restructure** via `CODEX_REFACTOR_PROMPT.md` — dataset-agnostic pipeline + per-aspect eval.
- [ ] **Regenerate clean PanoVGGT overlap_probe result** (argus-only runs overwrote the CSV/plots):
      `MODELS_TO_RUN=panovggt sbatch overlap_probe/slurm/probe_leonardo.slurm`.
- [ ] **Adopt PaGeR** as per-room geometry backbone (depth+normals); retire DAP.
- [ ] **Close the 0.913→0.842 connectivity gap** = detector RECALL: cubemap-face (undistorted) door
      detection + PaGeR geometric opening proposals. Simplest test: recall@15° cubemap vs ERP.
- [ ] Paper 1 remaining: real SfM/COLMAP anchor run (C5), ablation grid + hygiene retrain (G4/G5).
- [ ] Paper 2 prototype: GT-pose 2–3 rooms → per-room GS → off-the-shelf view-diffusion completion of
      one through-door transition (kill criterion first).

## Done (recent)
- Connectivity 0.913 GT / 0.842 detected (197 homes). overlap_probe: PanoVGGT degrades at low overlap,
  doorway-excepted. Flip-from-appearance = negative result. Argus integration inconclusive (not reported).

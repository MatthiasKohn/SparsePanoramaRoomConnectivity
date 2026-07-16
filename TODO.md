# TODO (open action items)

## DONE (2026-07-14)
- [x] DEPTH-GEN + exp29 pose/flip benchmark on Leonardo (trackD). 20 cyclic held-out homes.
- [x] exp33 flip diagnostic + oracle-pose test (trackE). **Result: flip prior is at chance,
      independent of depth (0.50 under GT-oracle pose).** C3 dropped, C4 reframed as the
      negative result, C5 promoted. See ResearchLog 2026-07-14 + NextStage C-table.

## NOW — critical path (layout rests on this after the flip negative)
- [ ] **REAL COLMAP HYBRID (C5 / G3):** the only thing that resolves the flip.
      `scripts/colmap_perspective.py` on 0025 + 2–3 more depth homes (perspective-split, already
      verified synthetically) → `exp24` COLMAP-coverage table (motivation: SfM registers X% of
      panos, links Y% of door adjacencies) → `exp27` real hybrid layout with the COLMAP anchors.
      Report layout error vs geometry-only. Scripts exist: `scripts/run_colmap_home.sh`,
      `scripts/run_hybrid_home.sh`.

## Optional — harden the flip negative (not a rescue, cheap)
- [ ] `python experiments/exp33_flip_diagnose.py --root $ZIND_ROOT --homes scripts/depth_homes.txt
      --raw_embed --oracle_pose --tag raw_oracle` — repeat oracle test with RAW DINOv2 features
      (no door head). If also ~chance → paper states neither task-tuned nor general-purpose
      features resolve the flip (preempts "your features are bad" objection).

## Parallel / independent (unaffected by the flip result)
- [ ] Train M2 distance head on data_floors (exp32) — `sbatch scripts/trackC_leonardo.slurm`;
      no depth needed. Compare val median dist vs exp31 DAP ~0.65 m. (PaperV2; also the honest
      "better metric depth" lever for layout.)
- [ ] **Argus (ECCV'26) baseline:** run released Argus on ZInD one-pano-per-room; show it degrades
      as inter-room covisibility → 0 while door-matching connectivity holds. Verify their overlap
      regime first (S C.4 "Scalability of Covisibility Module"). ResearchLog 2026-07-14.

## Writing
- [ ] Reframe Paper 1: connectivity (0.913/0.842, headline) + flip-is-a-fundamental-ambiguity
      (negative, C4) + hybrid SfM anchors fix layout (C5). Freeze numbers once COLMAP hybrid lands.

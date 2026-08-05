# SALVe integration — real estimated poses on ZInD

*Goal: replace GT poses with a real, published method (SALVe, ECCV 2022) for an honest
(pose_rmse, rot_err, door_gap, reconstruction) point vs the oracle. CoVisPose/Graph-CoVis were the
first choice but have NO public code/weights, so SALVe is the faithful option.*

## Method (why faithful, not hand-rolled)
SALVe: relative-pose hypotheses from W/D/O snapping -> a learned CNN **verifier** scores each -> GTSAM
**pose graph** gives global poses. The verifier is the contribution; we use SALVe's released verifier
weights + released MHNet W/D/O predictions, so we **train nothing**.

## Two decisions forced by reality
1. **Layout-only verifier (no depth).** SALVe's RGB modality needs HoHoNet depth, whose weights are
   gone from both official mirrors (Dropbox deleted, Drive 404). SALVe also released a rasterized-
   **layout** verifier (`6ac3f3e5...`, `modalities:["layout"]`) that needs no depth -> HoHoNet drops
   out entirely. Weaker modality than ceiling+floor RGB, but a legitimate released SALVe config.
2. **Must evaluate on a TEST-split building.** The verifier was trained on train+val. `0000` is in
   TRAIN -> leaky. Use a test-split building (0021, 0203, 0990, 0809, ...) AND run the oracle on the
   same one for a fair head-to-head. (Our oracle/noise/drift on 0000 remain valid — we train nothing.)

## Files
- `scripts/setup_salve.sh` — one-time: conda env (python+pip, then pip incl. gtsam/gtsfm/torch cu113),
  SALVe verifier ckpts (rgb 9fcbb628 / gtwdo b1198bad / **layout 6ac3f3e5**), MHNet predictions,
  vanishing angles. HoHoNet is attempted but non-fatal (not needed for the layout path).
- `scripts/run_salve.slurm` — one building: hypotheses -> layout BEV (no depth) -> verifier -> export.
- `salve_integration/gen_hypotheses_one_building.py` — hypotheses for ONE building (CLI only does splits).
- `salve_integration/export_salve_poses.py` — SALVe back-end -> `{stem:{rot_deg,pos}}` (`_est.json`)
  + a GT round-trip (`_gt.json`) that is the convention unit-test.

## Workflow
1. `bash scripts/setup_salve.sh` (login node). Env is done; this now also grabs the layout ckpt.
2. Set `BUILDING` (test split) + `FLOOR` in `run_salve.slurm`, then `sbatch scripts/run_salve.slurm`
   -> `salve_poses/<B>_<F>_{est,gt}.json`.
3. **Convention unit-test first**: `floor.py --pose_model salve --pose_file .../<B>_<F>_gt.json` -> pose RMSE ~0.
4. Real number: swap to `_est.json`. Run oracle on the SAME building. Both land in `results.csv`
   (pose_rmse, rot_err_deg, door_gap_m, PSNR/SSIM/LPIPS) -> directly comparable.

## Pose convention (derived from SALVe source; certified against door_gap=0)
SALVe's `generate_Sim2_from_floorplan_transform` puts poses in a RIGHT-handed frame via an
x-reflection of ZInD: its `Sim2.theta = -(our rot_deg)`, and its world pano position
(`Sim2.transform_from(0) = s*t`) is x-reflected vs ours. `export_salve_poses._pg_to_posefile`
converts back to OUR convention so `pose_c2w_gt` reproduces GT exactly:

    rot_deg = -theta ;  pos = (-world_x, world_y)   with world = Sim2.transform_from(origin)

Then `floor.py` applies a global Sim(3) (proper rotation) alignment for real-pose models before the
metrics (pose is only defined up to a similarity). ACCEPTANCE TEST: feeding SALVe's `_gt.json`
through `floor.py --pose_model salve` must give pose RMSE ~0 AND door_gap ~0. Verified on 0021
(0.000 m, 0.0 deg, 0.0 m over 7 doors). Only after that pass are the `_est.json` numbers trusted.

## First real result (0021 floor_01, layout-only verifier, CONF 0.5)
Oracle (GT): pose_rmse 0, rot 0, door_gap 0 (30/30 panos, 7 doors).
SALVe:       pose_rmse 1.82 m, rot 32.9 deg, door_gap 2.39 m (max 3.69), 18/30 localized, 3 doors.
-> layout-only SALVe is far from the oracle here; the large rotation error (33 deg) drives the door
   gap, consistent with the lever-arm finding. Caveats: one floor, weak (layout-only) modality,
   only 18/30 localized. Run more test floors for a distribution before concluding.

## First-run risk points (I cannot test these without the cluster)
- `test.py` dataloader reads `layout_data_root` (sed'd to our renders) over `--split`; only the one
  rendered building is present, so only it is scored. If it errors on missing buildings, we scope the
  dataloader or render a few test buildings.
- exact MHNet-prediction dir layout expected by `hnet_prediction_loader` may need a path tweak.
- unlocalized panos are dropped (no GT fallback); the run logs how many.

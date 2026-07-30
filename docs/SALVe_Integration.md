# SALVe integration — real estimated poses on ZInD

*Goal: replace GT poses with a real, published method (SALVe, ECCV 2022) to get an honest
(pose_rmse, rot_err, door_gap, reconstruction) point instead of synthetic noise. CoVisPose and
Graph-CoVis were the first choice but have NO public code/weights, so SALVe is the faithful option.*

## Why SALVe (and why faithful, not hand-rolled)
SALVe's front-end generates relative-pose hypotheses by snapping W/D/O midpoints between panos, a
learned CNN **verifier** scores each hypothesis, and a GTSAM **pose graph** assembles global poses.
The verifier is the whole contribution — skipping it (naive door-snapping) is NOT SALVe. We use the
released verifier weights + released MHNet predictions + HoHoNet depth, so we **train nothing**.

## Faithfulness caveats (state these honestly)
- **MHNet layout weights are not released**; we use their published MHNet *predictions* on ZInD.
- SALVe `run_sfm.py` never serializes poses (`PoseGraph2d.as_json` raises). Our
  `salve_integration/export_salve_poses.py` reproduces its exact reconstruction path and exports.
- Estimated poses are aligned to GT by Sim(3) before export (standard — pose is only defined up to a
  global similarity). Both our metrics are GT-relative, so this is correct.

## Files
- `scripts/setup_salve.sh` — one-time login-node: clone salve+HoHoNet, conda env (GTSAM/GTSFM/Open3D),
  download verifier ckpts + MHNet preds + HoHoNet ckpt + vanishing angles.
- `scripts/run_salve.slurm` — 5 stages: hypotheses → HoHoNet depth → BEV render → verifier → export.
- `salve_integration/export_salve_poses.py` — SALVe back-end → `{stem:{rot_deg,pos}}` (`_est.json`) +
  a GT round-trip (`_gt.json`).

## Workflow
1. `bash scripts/setup_salve.sh` (login node; the conda env is the main risk — GTSAM/GTSFM).
   Then set `CONFIG_NAME` in `run_salve.slurm` from `ls $SALVE_ROOT/salve/configs/` (must match the ckpt).
2. `sbatch scripts/run_salve.slurm` → `salve_poses/0000_floor_01_{est,gt}.json`.
3. **Convention unit-test** (must pass first):
   `python -m pipelines.floor --home $ZIND_ROOT/0000 --floor floor_01 --pose_model salve \
        --pose_file .../0000_floor_01_gt.json --depth_model gt_layout` → **pose RMSE ~0**.
4. Real number: same command with `_est.json`. One row lands in `results.csv` with pose_rmse,
   rot_err_deg, door_gap_m, PSNR/SSIM/LPIPS — directly comparable to the oracle and the noise sweep.

## Two front-ends (one ablation)
- `WDO_SOURCE=horizon_net` + MHNet verifier ckpt = the realistic end-to-end number.
- `WDO_SOURCE=gt` + GT-W/D/O verifier ckpt = isolates assembly/pose-graph error from detection error.

## Known first-run risk points
- conda solve (GTSAM/GTSFM) — see fallback in setup_salve.sh.
- target building must be in the processed `SPLIT` (default `test`); use `--building_id` where supported.
- exact per-stage flags / MHNet-prediction dir layout may need a one-line path tweak on first run.
- unlocalized panos are dropped (no GT fallback) — the run logs how many; large drop = caveat.

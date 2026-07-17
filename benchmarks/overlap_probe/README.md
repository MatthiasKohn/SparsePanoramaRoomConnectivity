# overlap_probe — do feed-forward panoramic reconstructors solve *our* setting?

A clean, self-contained experiment to test whether **Argus**, **PanoVGGT** (and a generic
**VGGT** baseline) actually recover camera poses when the capture is **one panorama per room**
(near-zero overlap except through doorways), or only when panoramas overlap.

## The design (one control does the work)
Run each model on the **same** ZInD homes under two regimes and compare to GT poses:
- `dense`  — every pano on the floor (high overlap; where these models are meant to shine).
- `sparse` — one room-central pano per room (**our** regime; almost all pairs are near-zero
  overlap — the "far" bin).

If a model keeps low error on `sparse`, it solves our problem and the door-pose pipeline is
obsolete. If error blows up on `sparse` while staying low on `dense`, that gap **is** the
contribution we build on.

## Metrics (gauge-robust — a model may return any frame/scale)
- **ATE (Sim3-aligned), meters + normalized** by scene diameter. The fitted scale `s` is
  reported; a truly *metric* model (Argus) should give `|log s| ≈ 0`.
- **Relative-rotation error** per pano pair (geodesic, deg). Relative rotations are invariant
  to the global frame, so this needs no alignment — the cleanest cross-model rotation metric.
- Both **stratified by GT overlap** (`same` / `adjacent` / `far`) so you see error rise as
  covisibility → 0.

## Run it
Validate the harness with zero external code (oracle must score ~0, noisy small non-zero):
```
python overlap_probe/run_probe.py --root $ZIND_ROOT --only scripts/depth_homes.txt \
    --models oracle,noisy --limit 8
```
Real run (after wiring adapters, below):
```
python overlap_probe/run_probe.py --root $ZIND_ROOT --only scripts/depth_homes.txt \
    --models argus,panovggt,vggt_tiled --out results/overlap_probe
```
Leonardo: `sbatch overlap_probe/slurm/probe_leonardo.slurm`.

## Wiring a real model (5 minutes each)
Each real adapter in `adapters.py` is a subprocess wrapper. You only change its `_cmd()` to the
model's actual inference call, and make that call write `out/pred.npz` with key `poses` of
shape `(N,4,4)` (camera-to-world, same order as the input images). Point the model at its repo
with an env var:
- Argus  → `ARGUS_DIR`  (metric; repo `RealseeTechnology/argus-realsee3d`)
- PanoVGGT → `PANOVGGT_DIR` (Sim3 scale)
- VGGT (tiled) → `VGGT_DIR` (perspective VGGT on e2p tiles of each pano; not metric)

Nothing else changes — `common.py`/`metrics.py`/`run_probe.py` stay fixed.

## Outputs
`results/overlap_probe/`: `probe_scenes.csv` (per scene/model row), `probe_overlap_strata.csv`,
`probe_dense_vs_sparse.png` (the headline), `probe_error_vs_overlap.png`.

## Files
`common.py` scene builder + GT · `overlap.py` covisibility proxy · `metrics.py` Umeyama/ATE/
rel-rot · `adapters.py` model interface + oracles + real-model stubs · `run_probe.py` CLI.

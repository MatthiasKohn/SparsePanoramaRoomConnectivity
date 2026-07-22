# Step-0 3D prototype — run recipe

**Goal.** Measure the disocclusion a Step-1 completion prior would have to fill: place per-room
panos + metric depth at **GT poses**, then look **through a doorway** at a neighbour room and
quantify how much of that view is unobserved. Kill-criterion first — no new model, GT poses,
off-the-shelf depth.

Pipeline: `pipelines/gs_room_prototype.py`. Reuses `sparsepano/gs/gsplat_init.py`
(`gaussian_init_from_pano`, `render_equirect`, `write_point_ply`) and `sparsepano/geometry/geom.py`.

## Key point: no gsplat, no GPU needed for Step-0
The disocclusion measurement is a **CPU numpy** point-reprojection + density metric. It runs
anywhere the fmodels venv (numpy + opencv) is available. gsplat/CUDA is only needed later
(surfel-accurate hole maps, or Step-1 differentiable optimisation) — see the bottom.

## Run the real Step-0 on Leonardo — ONE script, no login-node steps
Edit the CONFIG block at the top of `scripts/run_gs_prototype.slurm` (homes, floor, depth
source, stride) and submit:
```bash
sbatch scripts/run_gs_prototype.slurm
```
The pipeline is pure numpy/cv2 (no torch/gsplat/CUDA), so the whole thing runs inside the job —
nothing on the login node. It loops over the homes, writes each to
`$RESULTS_ROOT/gs_prototype/<tag>/`, and prints the convention check + disocclusion per home.
Set `DEPTH_SUB=dap_depth/depth_meters` to A/B DAP vs PaGeR depth. Homes with depth already
generated: `scripts/depth_homes.txt` (0053, 0070, 0023, 0032, …).

**Outputs** → `results/gs_prototype/<tag>/`: `input.png` (sanity: GS render from camera A —
should reproduce the pano), `novel.png` (view from the doorway into room B), `disocc.png`
(red = disoccluded cells), `merged.ply` (both rooms, GT-posed — open in a PLY viewer to eyeball
the geometry), plus a printed summary with the **DISOCCLUSION** fraction.

## Read the printed diagnostics FIRST
```
[zind] connected pair: <panoA> (room ..) <-> <panoB> (room ..) via door ..
[zind] convention check: az(A->roomB)=..  az(A->door)=..  disagree=.. deg  OK / !! FRAME MISMATCH
DISOCCLUSION : 0.xxx (n/N densely-seen cells under-sampled from novel view)
```
- **`convention check`** must say **OK** (disagree < 45°). ZInD `pose_c2w` and door
  `endpoints_xy` are in principle the same metric frame, but this was **not testable locally**
  (no ZInD panos on this machine). If it says **FRAME MISMATCH**, the world assembly is wrong —
  do not trust the disocclusion number; the fix is a sign/swap in how `build_room_gaussians`
  applies `pose_c2w` (cf. the `ZIND_CONV` swap in `sparsepano/doors/door_dataset.py`).
- **disocclusion** is the headline. Large (say >0.3) ⇒ the completion prior is the paper.
  Near-zero ⇒ naive unproject-splat already suffices and the interesting work is elsewhere.

## Local smoke test (already validated — no ZInD needed)
```bash
IM=".../Download_immersight_ 2026-06-23_10-23-49"
python -m pipelines.gs_room_prototype \
    --pano "$IM/panorama_1273530.png" \
    --depth "$IM/dap_depth/depth_meters/panorama_1273530.npy" \
    --baseline 2.5 --yaw 20 --stride 8 --tag smoke
```
Confirms init→world-transform→render→density-disocclusion→ply. The density metric responds to
viewpoint change (0.001 at a 0.3 m step → 0.858 at 2.5 m in a shallow room); it does **not**
saturate like a naive any-hit coverage test.

## Metric definition (why density, not "any hole")
A point splat has no surface, so far walls leak into would-be holes and every equirect direction
inside a room is filled → an "any point present?" test reads ~1.0 everywhere. Instead we
histogram Gaussian centres into coarse cells and threshold the count (`tau = tau_frac ×
median-nonempty-cell`). A cell is *well-observed* if it holds ≥ tau points; **disocclusion =
fraction of input-well-observed content cells that are under-sampled from the novel view**. This
is a CPU stand-in for the alpha/hole map a real gaussian rasteriser produces.

## gsplat surfel rendering (GPU) — the TRUE alpha hole-map
The CPU point metric is a density proxy; gsplat rasterises real surfels and its **alpha channel is
a true coverage/hole map**. Use it to check whether "coverage is solved" survives proper rendering.

1. ONE-TIME install (login node): `bash scripts/setup_gsplat.sh`
2. Run (GPU, no login-node steps): `sbatch scripts/run_gs_prototype_gpu.slurm`
   - Defaults to ONE home / ONE view (`--rasterizer gsplat`), so you can eyeball the auto-convention
     PSNR and `gs_novel_holes.png` before spending a sweep. Widen `HOMES`/`NOVEL_FRACS` after.
   - Outputs per tag: `gs_novel.png`, `gs_novel_alpha.png`, `gs_novel_holes.png` (red = true holes),
     `gs_input.png`. Prints: auto-convention PSNR (>18 dB = OK), input-view holes (control, ~0 if
     surfels fill), and the NOVEL through-door disocclusion (true alpha holes).
   - Renders a PERSPECTIVE view *through the doorway toward room B* (gsplat is pinhole), not equirect.

**First-run checklist (untested locally — no GPU here):** (a) auto-convention PSNR should be high
(>~22 dB); if it's low, the (basis×vflip) search or multi-room bleed is off — inspect `gs_input.png`.
(b) The perspective camera looks at room B's centroid; if it faces a wall, widen `GS_FOV` or set a
`--novel_frac` nearer the door. (c) gsplat JIT-compiles its kernel on first GPU import (minutes);
`TORCH_EXTENSIONS_DIR` caches it so later runs are instant.

### (Alternative install if the plain `pip install gsplat` route fails)
Use the prebuilt
wheel matching the stack to avoid a JIT compile on an offline node:
```bash
module load profile/deeplrn cineca-ai/4.3.0
env -u PYTHONPATH -u LD_LIBRARY_PATH $WORK/envs/fmodels/bin/python -m pip install \
    gsplat --index-url https://docs.gsplat.studio/whl/pt25cu124
# verify on a GPU node:
env -u PYTHONPATH -u LD_LIBRARY_PATH srun -p boost_usr_prod --gres=gpu:1 -t 00:05:00 \
    $WORK/envs/fmodels/bin/python -c "import torch,gsplat; print(torch.__version__, gsplat.__version__)"
```
If the wheel index has no `pt25cu124` build, fall back to a source build on the login node with
`CUDA_HOME` set (from the module) and `--no-build-isolation`. `env -u LD_LIBRARY_PATH` avoids the
recurring cineca-ai nvJitLink shadow.

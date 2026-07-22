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

## Run the real Step-0 on Leonardo (ZInD, where panos+depth+poses coexist)
```bash
# login or a CPU/GPU compute node; the fmodels venv only needs numpy+opencv
module load profile/deeplrn cineca-ai/4.3.0
source $WORK/envs/fmodels/bin/activate
export PROJECT_ROOT=$HOME/projects/SparsePanoramaRoomConnectivity   # adjust
export ZIND_ROOT=$WORK/data/zind/full_dataset                        # adjust
cd $PROJECT_ROOT

PYTHONPATH=$PROJECT_ROOT python -m pipelines.gs_room_prototype \
    --home $ZIND_ROOT/0053 --floor floor_01 \
    --depth_sub pager_depth/depth_meters \
    --stride 4 --render_h 512 --tag zind_0053
```
Swap `--depth_sub dap_depth/depth_meters` to compare DAP vs PaGeR depth on the same view.
Homes with depth already generated: `scripts/depth_homes.txt` (0053, 0070, 0023, 0032, …).

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

## Optional upgrade — gsplat (torch 2.5.0 + cu124) for surfel-accurate holes / Step-1
Only if you want true surfel occlusion or differentiable optimisation (`sparsepano/gs/gs_optim.py`).
Install into the fmodels venv from the login node (compute nodes are offline); use the prebuilt
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

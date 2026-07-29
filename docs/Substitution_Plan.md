# Substitution study — plan, protocol, metrics

*The experimental spine of the paper: build the floor with all-GT (the oracle upper bound), then
swap ONE block at a time for a baseline and measure how far the reconstruction falls. Everything
runs through one pipeline (`pipelines/floor.py`) with swappable providers, so results are directly
comparable and land in one `results.csv`.*

## Swappable blocks (providers)
| Block | flag | GT (oracle) | Baselines |
|---|---|---|---|
| Poses | `--pose_model` | `gt` | `noise` (--noise_deg/--noise_m), `covispose`, `salve`, `badgr` |
| Depth/geometry | `--depth_model` | `gt_layout` (or `fused`) | `pager`, `dap` |
| Connectivity | `--connectivity` | `gt` | `detected` (TODO) |
| Completion | `--completion` | `none`/`cv2` | `sd` (SDXL), Pano2Room (later) |

Each provider lives in `sparsepano/providers/{poses,depth,connectivity,completion}.py`.

## The one rule for comparability
**Exactly one block leaves GT per run; all others stay GT.** Same homes/floors, same held-out
panos, same render/opt settings. That isolates each component's contribution to the drop.

## Metrics (written per run to results.csv)
Reconstruction quality (held-out novel view — build from 1 pano/room, score the rest):
- **PSNR / SSIM / LPIPS** on covered pixels (LPIPS needs the `lpips` pkg cached).
- **coverage** = fraction of the held-out view actually observed (the rest is disocclusion).

Pose quality (for pose swaps):
- **pose_rmse_m** = Umeyama-aligned camera-centre RMSE vs GT.
- **rot_err_deg** = mean camera rotation error vs GT.

`results.csv` columns: `home, floor, pose_model, depth_model, connectivity, completion,
noise_deg, noise_m, pose_rmse_m, rot_err_deg, n_rooms, n_eval, coverage, psnr, ssim, lpips`.

## The experiments (run order)
1. **Oracle**: `--pose_model gt --depth_model gt_layout` (and a `fused` variant). The upper bound.
2. **Pose sensitivity** (no external code): `--pose_model noise` swept over
   `--noise_deg {5,10,15,20,25} --noise_m {0.1,0.2,0.3,0.5}` (a couple of seeds). Plot
   PSNR/LPIPS vs `pose_rmse_m` -> the headline "how much pose error can the 3D tolerate" curve.
3. **Depth swap**: `--depth_model pager`, then `dap` (pose=gt). Cost of monocular depth vs layout.
4. **Real pose methods**: run CovisPose -> BADGR -> SALVe on the floor, dump `stem->{rot_deg,pos}`
   to a json, pass `--pose_model <m> --pose_file <json>`. Each lands as a POINT on the curve from (2).
5. **+Completion / connectivity**: measured as the delta vs the matching no-completion / GT-conn run.

## What's implemented vs pending
- **Now**: `gt`, `noise` poses; `gt_layout/fused/pager/dap` depth; `gt` connectivity; `none/cv2/sd`
  completion; held-out metrics + pose error -> `results.csv`; room-aware rendering (see-through fix);
  optional `--visuals` (ply + walkthrough.mp4).
- **Pending integration** (each is its own step): CovisPose/SALVe/BADGR pose dumps (`--pose_file`);
  `connectivity=detected`; Pano2Room-style 3D-consistent generation.

## Run
```
sbatch scripts/run_floor.slurm     # edit POSE_MODEL/DEPTH_MODEL/COMPLETION/EXTRA in the CONFIG block
```
Metrics-only runs are fast (drop `--visuals`); add `--visuals --optimize` for the ply + walkthrough.

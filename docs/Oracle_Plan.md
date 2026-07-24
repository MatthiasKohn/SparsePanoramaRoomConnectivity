# Plan — GT oracle upper bound → top-down substitution study

*The forward experimental plan. Build the best-possible floor with every ground truth, then
knock out one GT component at a time and measure how far the result falls. The component whose
removal hurts most = where our contribution should go. Implementation now = the oracle run only
(`pipelines/oracle_floor.py`); the substitutions are designed here but built later.*

## Why this design
Instead of guessing which part (pose? geometry? completion?) is the bottleneck, we **measure**
it. A single upper bound + one-variable ablations turns "what should we work on" into a table of
Δ-from-oracle. It also directly answers the reviewer question "what if you just use an existing
pose estimator" — we show exactly how much that costs.

## The pipeline components (each is a substitution axis)
| Component | GT (oracle) | Baseline substitute (later) |
|---|---|---|
| Camera poses | ZInD floor_plan_transformation | SALVe / **BADGR** (verify it runs at our overlap) |
| Per-room geometry / depth | **GT room layout → depth** (`layout_depth.py`) | PaGeR (monocular) → then DAP |
| Room appearance | one real pano per room | (kept real; input modality) |
| Connectivity / which rooms | all GT rooms of the floor | detected doors → our connectivity graph |
| Completion of unseen regions | **none yet** (stage 2a) | off-the-shelf view-diffusion / pano inpainting |

## Stage 2a — the oracle floor (BUILD NOW)
`pipelines/oracle_floor.py`. Per pano: depth from the GT layout (vertical-prism ray-cast, aligned
to the pano via the door-azimuth convention), unproject to 3D Gaussians, place at the GT pose;
merge all rooms → one floor (3DGS). **No generation** — holes stay holes (measured, not filled).

Validated so far: layout-depth is exact (doors sit on walls at their GT distance); a 6-room floor
assembles to a plausible ~8×10 m footprint, 2.45 m ceiling.

## Evaluation protocol — FROZEN (identical for oracle and every substitution)
Chosen 3(c): always report the view-quality set AND the geometric set, so any run is comparable.

- **(A) Held-out novel view.** ZInD rooms with ≥2 panos: rebuild the floor WITHOUT one pano,
  render perspective tiles (yaw sweep) at its GT pose, score vs that pano's real `e2p` tiles →
  **PSNR / SSIM / LPIPS** on covered pixels. (Target setting stays one-pano-per-room; the extra
  panos are only the measuring stick.)
- **(B) Geometric.** **coverage / disocclusion** = fraction of the held-out view with no surfel
  (alpha < τ) — what a completion prior would have to fill.
- **(C) Qualitative.** through-floor **walkthrough** frames + the merged `floor.ply`.
- **Pose error harness.** `pose_rmse` (Umeyama camera-centre RMSE) = 0 in the oracle; it becomes
  the x-axis when we substitute poses.

Report per room, per floor, and a mean. Dataset: ZInD, a handful of homes/floors with connected
multi-pano rooms. Representation: 3DGS (gsplat).

## Substitution roadmap (DESIGN ONLY — build after the oracle lands)
Run each with **all other components at GT**, so the Δ isolates one variable.

1. **Pose: GT → BADGR/SALVe.** Suspected biggest lever (Step-0 showed coverage is easy under GT
   poses; pose is the open question). Plot metric vs `pose_rmse`. **Risk:** BADGR assumes covisible
   walls — confirm it even produces poses in our near-zero-overlap regime; if not, that *is* a finding.
2. **Geometry: GT layout-depth → PaGeR (then DAP).** How much does monocular depth cost vs true
   layout? Isolates the geometry component (our A/B said PaGeR≠better for distance — test end-to-end).
3. **Connectivity: GT rooms → detected doors + our graph.** Wrong/missing edges place rooms wrong;
   measures how connectivity recall propagates to 3D.
4. **+ Generation.** Add off-the-shelf completion on top of the (GT-pose) floor; measured as the
   **gain** in coverage/LPIPS over stage 2a. This is the "last mile", quantified.

The ordering (pose → geometry → connectivity → generation) front-loads the components we suspect
matter most; the table of Δ-from-oracle localizes the contribution objectively.

## Kill / decision criteria
- If a substitution barely moves the metrics, that component is *not* our lever — drop it.
- If pose substitution collapses the result, the contribution is pose/registration (our done work
  becomes the enabler, exactly the pitch).
- If holes stay large even at GT everything, generation is essential and becomes the headline.

## What is implemented now vs later
- **Now:** `pipelines/oracle_floor.py` (assembly + frozen eval harness + walkthrough), the
  `layout_depth.py` geometry oracle, `scripts/run_oracle_floor.slurm`. Stage 2a, no generation.
- **Later:** each substitution wires a different component into the SAME `build_floor` + eval
  (e.g., `--poses badgr`, `--depth pager`), plus the generation step.

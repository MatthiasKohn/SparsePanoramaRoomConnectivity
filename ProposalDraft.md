# Whole-Floor 3D from Sparse Panoramas — one-page research plan (draft)

## Problem
**Input (test time):** a sparse set of gravity-aligned 360° panoramas — roughly one per
room of an *unknown* floor — with near-zero visual overlap except *through doorways*. No
poses, no floor plan.
**Output:** a single globally consistent **3D reconstruction** of the whole floor
(dense geometry + appearance, renderable / navigable), together with the room-connectivity
graph and camera poses that scaffold it.

## Why it is hard (the scientific core)
Standard SfM fails: with near-zero overlap there are almost no cross-view correspondences
except at doorways. Each door correspondence fixes a relative pose only up to the
*which-side* flip. Monocular depth gives metric scale but is noisy. And every room is seen
from essentially **one** viewpoint, so per-room 3D is a single-view (ill-posed) problem.
The task is to fuse locally-ambiguous, low-overlap door cues into one coherent 3D floor.

## Positioning — what is and isn't novel (honest)
- **2D layout from sparse panoramas is SALVe** (ECCV'22): it infers windows/doors/openings,
  hypothesizes pairwise adjacency, a learned verifier resolves the alignment ambiguity, a
  pose graph (GTSAM) gives room poses, HorizonNet layouts are stitched into a **2D floor
  plan**. Our connectivity + flip-resolving pose graph solve the *same* 2D sub-problem — so
  we do **not** claim it as the contribution; we use it as infrastructure.
- **Single-room panoramic Gaussian splatting is also crowded** (360-GS, Splatter-360,
  PanoPlane): these reconstruct one room / wide-baseline views *with overlap*.
- **The gap neither addresses:** a *multi-room, whole-floor* 3D reconstruction from
  one-pano-per-room with **near-zero overlap**, where rooms are linked only through
  doorways and poses are unknown/ambiguous. SALVe stops at 2D; panoramic-GS assumes a single
  space with overlap. Bridging them is the contribution.

## Contributions (claim)
1. **Whole-floor 3D from sparse, door-connected panoramas:** place per-room panoramic
   Gaussian reconstructions into one global 3D model using a learned door-matching pose graph.
2. **A learned through-door appearance prior** that resolves the which-side flip *inside* the
   pose graph (resolves bridge edges that cycles cannot) — enabling correct 3D stitching
   across doorways.
3. **A floor-level evaluation** of multi-room reconstruction quality (not just 2D layout).

## Pipeline
panoramas → (per pano) door detection + monocular depth → contrastive cross-view **door
embedding** (matching ⇒ connectivity + correspondences) → door-anchored **relative pose** per
edge (+ flip candidate) → **SE(2) pose graph** (cycles + appearance prior resolve flips) →
per-room **panoramic GS** initialized from metric depth → **fused global 3D floor**, with a
joint photometric/feature refinement across shared doorway regions.

## Experiment plan (staged; front-loads the 3D test you want)
- **E0 — single-room GS sanity (now):** init Gaussians from one pano's metric-depth point
  cloud; optimize on the equirect image; render at the input pose (reconstruction check) and
  at small offsets (geometry plausibility). Validates the per-room 3D building block.
- **E1 — two-room through-door (GT pose):** fuse two rooms' splats via GT relative pose;
  render a novel view *through the doorway*; measure coherence. Tests cross-room stitching in
  isolation from pose error.
- **E2 — two-room with estimated pose + flip:** replace GT with the door-anchored pose + the
  appearance prior; show the correct flip yields a coherent 3D join, the wrong flip does not.
- **E3 — whole floor:** pose graph over all rooms → fused 3D floor on held-out ZInD homes.
- **E4 — refinement:** joint photometric/feature (DINO) loss over doorway-overlap regions to
  polish pose + geometry (the exp16 finding says: structural/feature loss, multi-view, not
  naive single-warp RGB).

## Metrics & baselines
- **3D / rendering:** novel-view PSNR/SSIM/LPIPS at held-out doorway views; depth/geometry
  consistency across the doorway; qualitative walkthroughs.
- **Scaffold:** connectivity AP, pose/position error, **flip accuracy** (embedding vs cycles
  vs random), layout error vs #rooms.
- **Baselines:** SALVe (2D layout, as the pose/connectivity reference); geometry-only +
  random-flip; single-room panoramic-GS placed by GT vs by our graph.

## Assumptions & risks (state up front)
- Gravity alignment (yaw-only) — true on ZInD; deployment needs IMU / vanishing-point rectify.
- One-pano-per-room, connectivity-through-doors.
- **Biggest risk:** single-view per-room GS is under-constrained → leans entirely on monocular
  depth + priors; naive photometric is unreliable on bland walls (our exp16). Mitigate with
  depth-regularized GS, feature losses, and feed-forward priors (à la Splatter-360).
- GT is used for *training supervision only*; the test-time system must run from panoramas
  (move door matching from GT bearings to the embedding).

## Related work to cite
SALVe (ECCV'22); ZInD; 360-GS; Splatter-360; PanoPlane; HorizonNet; DINOv2; 3D Gaussian Splatting.

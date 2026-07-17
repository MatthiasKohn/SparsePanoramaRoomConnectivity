# Project Overview — Sparse-Panorama Room Connectivity

*The single "read first" doc. Problem, hypothesis, current pipeline, status, headline results.
Deeper detail: `ResearchLog.md` (evidence), `Roadmap.md` (plan, both papers), `RelatedWork.md`.*

## Problem
From a **sparse set of gravity-aligned 360° panoramas — roughly one per room, with near-zero
visual overlap except through doorways** — recover:
1. the **room-connectivity graph** (which rooms connect, via which doors),
2. **metric relative camera poses / SE(2) floor layout**, and
3. (later) a navigable **3D reconstruction** of the whole floor.

No poses, no floor plan at test time. PhD context: 3D scene perception & change detection.

## Why it is hard
Classical SfM needs cross-view correspondences; at near-zero overlap there are almost none except
at doorways. Each door correspondence fixes relative pose only up to a **which-side flip**.
Monocular depth gives metric scale but is noisy. Each room is seen from ~one viewpoint, so per-room
3D is single-view (ill-posed). The task fuses locally-ambiguous, low-overlap door cues into one
coherent floor.

## Core hypothesis
**Doorways are geometric apertures.** Pixels inside a doorway are rays into the neighbouring room;
the surfaces seen through the door are potentially visible from both rooms, so the doorway is the
one shared structure that carries a cross-room constraint.

## Current pipeline / track
```
panos → per-room metric geometry (depth + normals; foundation model, e.g. PaGeR)   [FM, not our contribution]
      → door detection + cross-view door embedding
      → connectivity graph + door-anchored relative pose (flip resolved by embedding + geometry)   [OUR contribution]
      → global SE(2)/SE(3) layout
      → (Paper 2) NoPoSplat/CAT3D-style feed-forward 3DGS + diffusion, conditioned on connectivity,
        to place non-overlapping rooms and hallucinate through-door + unseen regions → navigable floor
```
Positioning in one line: **foundation models now solve panoramic geometry (PaGeR) and
dense/overlapping reconstruction (Argus, PanoVGGT, IM360, NoPoSplat); the open problem — and our
lane — is registration + completion of a multi-room floor from one pano per room, where covisibility
collapses to the doorway.**

## Status snapshot (2026-07)
**Done / validated**
- **Connectivity** (our anchor result): assign-AP **0.913 with GT doors, 0.842 with detected doors**,
  197 scene-disjoint held-out ZInD homes. Cross-view door embedding (DINOv2 + head, InfoNCE, same-home
  hard negatives) → global 1-to-1 door assignment.
- **overlap_probe** finding: PanoVGGT (SOTA feed-forward panoramic) degrades sharply as overlap→0
  (dense relRot 5.7° → sparse 23.9°; within-scene strata 1.7°/6.9°/17.2° for same/adjacent/far),
  **doorway-adjacent pairs excepted** — the door is where residual pose signal survives.

**Negative / settled**
- **Which-side flip from appearance is a genuine geometric ambiguity** (exp29/33): embedding flip
  accuracy ≈ chance (0.45) at scale, even under GT-oracle pose. Reported as a rigorous negative;
  it *necessitates* metric anchors. Stop tuning the flip prior.
- **Argus** could not be reliably applied to ZInD panoramas (rig-coupled geometry) — not reported.

**Next levers** (see Roadmap): close the 0.913→0.842 gap via higher-recall **door detection on
undistorted cubemap faces + PaGeR geometric opening proposals**; adopt PaGeR as the geometry backbone;
report a high-precision operating point + end-to-end layout error, not AP alone.

## Success criteria
Robust to sparse overlap · works on realistic indoor scenes · scales beyond hand-picked examples ·
pluggable across datasets (ZInD primary; abstraction ready for others).

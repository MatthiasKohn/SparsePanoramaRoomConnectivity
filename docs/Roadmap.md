# Roadmap — Paper 1, Paper 2, open questions

*Merges the former NextStage (Paper 1), PaperV2Plan (Paper 2), OpenQuestions, and the July
Direction decision. Forward-looking; evidence lives in `ResearchLog.md`.*

---

## Paper 1 — Connectivity + globally consistent SE(2) floor layout

**Scope call:** Paper 1 = connectivity + SE(2) layout. 3D/GS is a *demo* section, not a claim
(the GS work is init-only and the panoramic-GS field is crowded).

**Contribution — one learned door embedding does three jobs:**
1. **matches** doors across opposite sides → room-connectivity graph;
2. exposes the **which-side flip** as a genuine geometric ambiguity (rigorous negative result);
3. its match confidence **weights** edges for a robust SE(2) pose graph (+ optional flip-free SfM anchors).

**Claims & status**
| # | Claim | Status |
|---|-------|--------|
| C1 | Connectivity from panoramas alone: assign-AP 0.913 (197 homes, random ~0.31); large-home precision fixed by same-home hard negatives (0.55→0.86) | DONE |
| C2 | Connectivity-as-global-door-assignment beats independent pair scoring (0.737→0.913 with hard-neg+unfreeze) | DONE |
| C3 | Which-side flip from appearance ≈ chance at scale (0.45; 0.50 under GT pose → not depth-limited) | DONE → NEGATIVE |
| C4 | Real floors are trees (no cycles) so cycle-consistency can't fix flips either → flip is an unbroken ambiguity, necessitating metric anchors | DONE (report as analysis) |
| C5 | Hybrid pose graph: 15–30% flip-free metric anchors cut layout error ~6× | mechanism real (exp27); needs a real COLMAP/SfM run |
| C6 | Wrong flip breaks 3D: 0% vs 52% interpenetration | DONE (demo) |

**Open gaps to a submission**
- **G1 — detector-driven front end (largest lever):** headline is 0.842 with detected doors; gap is
  **detector recall** (P/R @15° ≈ 0.31/0.66). Close via cubemap-face (undistorted) detection + PaGeR
  geometric opening proposals for leaf-less openings. Report AP(detected) with AP(GT) as oracle upper bound.
- **G3 — hybrid/SfM anchor (C5):** needs a real spherical/perspective-split SfM run (COLMAP rejected
  "SPHERE"; use 8 perspective ring views). COLMAP-alone coverage is also the motivation table.
- **G4/G5 — hygiene + ablations:** clean hard-neg retrain with early-stopping; the 2×2×scoring grid,
  scaling curve, crop-fov, region-masking mechanism probe.
- (G2 — multi-floor pose/flip benchmark — DONE via exp29.)

**Evaluation protocol (frozen)**
- Scene-disjoint splits at home level; `runs/hardneg/val_homes.txt` (197) is the frozen test set.
- Test-time input = panoramas only; GT only for training pairs + eval labels; headline uses detected doors.
- Metrics: connectivity mean AP + P/R/F1 at operating point, stratified by #rooms and connection type
  (opening/open/closed door); flip accuracy (bridge vs cycle); layout error (m, after Umeyama), reported
  as a distribution; detector P/R @15°; 3D-demo interpenetration %. Always show chance. **Add: a
  high-precision operating point and end-to-end layout error, not AP alone** (deployment relevance).
- Baselines: SALVe (reported numbers), geometry-only + free-space (our ablations, both fail — the point),
  COLMAP-alone coverage, VGGT/PanoVGGT qualitative.

---

## Paper 2 — Generative floor completion (the chosen forward track)

**Core bet:** foundation models reconstruct what was *observed*; with one pano per room most of the
floor is *unobserved*. The defensible problem is **generative completion of the unseen floor +
through-door transitions**, with **connectivity as the structural prior** enforcing cross-room
consistency. Novelty is the SETTING (multi-room, one-per-room, doorway-connected), not "another NVS
diffusion" (crowded: CAT3D, ReconFusion, Pano2Room, ZeroNVS).

**Pipeline:** sparse panos → per-room metric geometry (PaGeR) → connectivity + door-anchored poses
(Paper 1) → per-room 3DGS (holey) → **diffusion prior conditioned on partial render + connectivity/
doorway constraints** hallucinates unobserved regions & through-door views → distill back into GS
(CAT3D/NoPoSplat-style) → refine → navigable floor.

**Staged experiments (kill criteria — do NOT build everything at once):**
1. Minimal prototype: take **GT poses** on 2–3 rooms (skip pose), per-room GS, off-the-shelf
   view-diffusion conditioned on a partial render; test completion of one room + one through-door
   transition. Kill if the diffusion can't plausibly complete a single through-door view.
2. Add connectivity conditioning; test cross-room consistency vs unconditioned.
3. Scale to whole floor; evaluate novel-view PSNR/SSIM/LPIPS at held-out doorway views + walkthroughs.

**Architecture template:** NoPoSplat (pose-free, canonical-anchor, feed-forward 3DGS, intrinsics-as-
token) is the blueprint; the novelty is making it work at zero overlap via the door/connectivity structure.

---

## Open questions (live)
- **Detection recall** is the current bottleneck for the deployable connectivity number — does cubemap-
  face detection + PaGeR opening proposals close the 0.913→0.842 gap? (simplest test: recall@15° on
  cubemap vs ERP, same held-out homes).
- Can geometry-in-the-loop (pose-graph consistency) re-score/reject false door matches to raise precision?
- Is 0.913 "enough"? Research: yes. Deployment (Immersight): needs a high-precision operating point +
  human verification — that's a deployment gap, not a research flaw.
- (Settled) *How much overlap is required* — answered by overlap_probe: feed-forward pose collapses as
  overlap→0, doorway-excepted.
- Future: change-detection representation (geometry vs geometry+semantics; temporal consistency).

## Immediate next actions
1. Repo restructure (see `../CODEX_REFACTOR_PROMPT.md`): dataset-agnostic pipeline + per-aspect eval.
2. Regenerate the clean PanoVGGT overlap_probe result (argus-only runs overwrote the CSV/plots).
3. Adopt PaGeR as the per-room geometry backbone (depth+normals); retire DAP.
4. Cubemap-face door detection recall experiment (G1 lever).

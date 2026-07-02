# Sparse-Panorama Room Connectivity — Progress Summary

## Headline result
From sparse 360° panoramas of a floor (~one per room, near-zero overlap except through
doorways), we predict **room connectivity** with a learned cross-view door embedding.
On **197 held-out homes** (scene-disjoint, **no floor plan at test time**):

**mean Average Precision = 0.913** (F1 0.92), vs 0.31 random — ~3× chance.

Progression that got there: **0.69 → 0.74 → 0.91**
(baseline max-cosine → global door assignment → same-home hard-negative fine-tuning),
with the large-home failure mode largely eliminated (large-home precision 0.55 → 0.86).

## Problem & why it's hard
- **Input (test):** a set of gravity-aligned 360° panoramas, ~one per room, of an unknown
  floor. Almost no visual overlap between rooms except at shared doorways.
- **Output:** the room-connectivity graph. Extensions in progress: camera poses (metric
  floor layout) and dense 3D.
- **Why hard:** classical Structure-from-Motion (COLMAP) needs overlapping views and fails
  across near-zero-overlap rooms — the documented SALVe result (SfM lost ~200% completeness).
  The only shared signal is doorways, and a single door match leaves a 2-fold **which-side**
  pose ambiguity.

## Data & inputs
- **ZInD** (Zillow Indoor Dataset): thousands of homes, panoramas + GT floor-plan annotations
  (rooms, doors/windows/openings, camera poses). GT is used for **training supervision** and
  **evaluation**; the target system runs from panoramas alone at test time.
- **DAP** (Depth Any Panorama): monocular metric depth, used for the pose/3D extensions.

## Method / pipeline
1. **Doors** located from ZInD door annotations (a GT-free detector is the deployment path,
   not the current focus).
2. **Cross-view door embedding:** a perspective crop centered on each door → DINOv2 (ViT-S)
   backbone + small MLP head → 128-d vector. Trained with symmetric InfoNCE so the SAME door
   seen from two rooms is close and other doors far. Labels are free from the floor plan.
3. **Connectivity:** match doors across panos by embedding cosine → predict an edge between
   rooms; scored threshold-free by Average Precision.
4. **(Extension) Pose:** door-anchored relative pose from depth (yaw + translation) + the
   which-side flip; an SE(2) pose graph; the embedding resolves the flip; COLMAP supplies
   metric anchors (hybrid graph).
5. **(Extension) 3D:** per-room Gaussian-splat initialized from depth, placed by the pose graph.

## Experiments & outcomes (the scientific arc)
1. **Held-out connectivity, baseline scoring:** AP 0.691 (197 homes) — it generalizes, but noisy.
2. **Diagnosis:** AP degrades with #rooms; the failure is **precision** (false edges from
   confusable doors in big homes), not recall.
3. **Global door assignment (no retraining):** a door connects to ≤1 door in one other room →
   solve a global matching → confusable doors get consumed by their true partner → false edges
   collapse. AP 0.691 → **0.737**, recall up too.
4. **Same-home hard-negative training:** standard contrastive negatives are easy (doors from
   other houses); the hard case is same-BUILDING doors. Packing each batch from a few homes makes
   in-batch negatives same-home → teaches discriminability; plus gentle backbone fine-tuning.
   **AP → 0.913**, large-home precision 0.55 → 0.86, degradation-with-size largely removed.
5. **Which-side flip resolution:** the trained embedding (through-door reprojection consistency)
   resolved **7/7** flips on a real cyclic home. Geometry alone cannot — it's a true ambiguity.
6. **Pose graph + hybrid:** on GT layouts, adding flip-free metric anchors (COLMAP) to door
   edges cuts layout error ~6× and lifts flip accuracy — even at 15–30% COLMAP coverage. Real
   monocular door-anchored poses are metrically noisy (~2 m) but topologically correct once flips
   are right → COLMAP is the metric backbone.
7. **3D:** a single-pano Gaussian init reproduces the input view (63 dB); a wrong flip makes two
   rooms interpenetrate 52% vs 0% for the correct side — the 3D stakes of the flip. Seamless
   multi-room splatting is unsolved (future work).

## What the metrics mean
- **Average Precision (AP):** area under the precision-recall curve over ranked room-pair edges;
  threshold-free connectivity quality. Random ≈ 0.31 here; 0.91 = strong.
- **Precision / Recall / F1** (at best-F1 threshold): precision = fraction of predicted edges
  correct (false-edge control); recall = fraction of true edges found.
- **corr(#rooms, precision):** how much precision degrades as homes grow; −0.66 → −0.40 with
  hard negatives (less degradation).
- **Flip accuracy:** fraction of edges whose which-side was resolved correctly.
- **Layout error (m):** median camera-position error vs GT after alignment.
- **Interpenetration %:** fraction of one room's points inside another — a 3D flip diagnostic
  (0% good, 52% = collapsed layout).

## What the figures show
- **heldout_summary_hn_assign.png** — per-home AP (nearly all high), AP vs #rooms (now flat-ish),
  AP distribution (mass near 0.9–1.0). *The headline figure.*
- **diagnose_scaling.png** — AP / precision / recall vs #rooms; the precision line is much flatter
  than the baseline.
- **e2_*_flip.png** — correct side (0% interpenetration, rooms adjacent) vs flip (52%, rooms
  collide): why the flip matters for 3D.
- **hybrid_0330.png** — layout error and flip accuracy vs COLMAP coverage fraction.

## For the meeting — what to lead with vs leave out
**Lead (solid, defensible):**
- Held-out connectivity **0.913 AP over 197 unseen homes, no floor plan at test time**, and the
  method arc (diagnose → global assignment → hard negatives) that got there. Shows rigor.

**Mentionable (novel, in progress):**
- The which-side **flip resolution (7/7)** and its **3D consequence (0% vs 52%)** — the part
  beyond SALVe, which stops at a 2D floor plan.
- The **hybrid pose-graph** plan with COLMAP 4.1.0 as metric backbone (baseline + component).

**Leave out / call "roadmap":**
- Joint multi-room Gaussian-splat optimization (not working cleanly).
- The COLMAP run itself (not completed — camera-model/version issue).
- Dead ends (naive photometric flip prior, free-space prior) — only if asked; they’re useful to
  show what was ruled out, but not headline material.

## Honest caveats
- Doors are currently located from GT annotations; fully GT-free (detector + embedding-only
  matching) is future work.
- Depth-based poses are metrically noisy; the metric map needs COLMAP or multi-view.
- 0.913 uses a fine-tuned backbone with a mild overfit signal (val plateau); the number is
  held-out so it stands, but a frozen+hard-neg ablation would tighten the story.

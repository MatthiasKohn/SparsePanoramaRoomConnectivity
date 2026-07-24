# Landscape Matrix — where the white space is (novelty map)

*Purpose: answer "is what we plan already done?" by mapping the field on capability axes, not by
listing papers. Read the two grouped tables, then the white-space verdict. Grounded in
`RelatedWork.md` + verified sources (BADGR project page, Extreme-SfM repo — July 2026).*

## Axes
- **Input:** perspective (P) vs panorama (360).
- **Overlap regime:** `dense` · `wide-baseline+covis` (sparse but walls/features co-observed across
  views) · `near-zero` (overlap only through doorways — OUR regime) · `none/one-per-room`.
- **Sparsity:** images per room.
- **Pose:** needs GT poses / **estimates** pose / pose-free feed-forward.
- **Outputs:** Conn = explicit connectivity graph · Pose/2D = metric poses or 2D floor plan ·
  3D = 3D geometry · RGB = appearance/renderable · **Gen = generative completion of *unseen* regions.**

Legend: ✓ yes · ~ partial/implicit · – no.

---

## Group A — registration / layout from sparse panoramas (the "pose" side)

| Method (year) | Input | Overlap regime | Imgs/room | Pose | Conn | Pose/2D | 3D | RGB | Gen |
|---|---|---|---|---|---|---|---|---|---|
| **Extreme SfM** (ICCV'21) | 360 | **none / one-per-room** | ~1 | estimates | ~ | ✓ (coarse: <1 m for 47% top-1) | – | – | – |
| **SALVe** (ECCV'22) | 360 | wide-baseline+covis (W/D/O) | sparse | estimates | ~ | ✓ | – | – | – |
| **CovisPose** (ECCV'22) | 360 | wide-baseline+covis | sparse | estimates | – | ✓ (coarse init) | – | – | – |
| **Graph-CoVis** (CVPR'23-W) | 360 | wide-baseline+covis | sparse | estimates | ~ | ✓ | – | – | – |
| **BADGR** (CVPR'25, HL) | 360 | wide-baseline+**covis walls** | ≥0.6, "up to 30" | **refines** coarse | – | ✓ (SOTA: 12 cm) | – | – | ~ (2D *layout* inpaint only) |
| **→ OURS (registration)** | 360 | **near-zero (doorway)** | **~1** | estimates | **✓ (first-class)** | ✓ (SE(2), door-anchored) | – | – | – |

**Reading:** the pose side is **crowded**. BADGR is the accuracy leader (12 cm) but is a *refiner* on
top of a covisibility-based coarse init and assumes **walls co-observed across images** — likely more
overlap than our one-pano-per-room regime. Only **Extreme SfM** explicitly targets our zero-overlap,
one-per-room setting, and it is coarse and Manhattan-restricted. Our differentiators here are narrow
but real: an **explicit connectivity graph** as output, the **flip-necessity negative result**, and
the near-zero regime. None of these is a big-3D-paper on its own.

---

## Group B — overlap-dependent 3D foundation models (need covisibility; not completion)

| Method | Input | Overlap regime | Pose | 3D | RGB | Gen | Note |
|---|---|---|---|---|---|---|---|
| **PanoVGGT** (CVPR'26) | 360 | dense→degrades near-zero | feed-fwd | ✓ | ✓ | – | our probe: collapses as overlap→0 (evidence) |
| **Argus** (ECCV'26) | 360 | dense/covis | feed-fwd | ✓ | ✓ | – | rig-coupled; didn't transfer to ZInD |
| **IM360** (ICCV'25) | 360 | dense | SfM | ✓ | ✓ | – | 360-native SfM+mesh+texture |
| **DUSt3R/MASt3R/VGGT** | P | dense/covis | feed-fwd | ✓ | ~ | – | fail in low-overlap multi-room (rooms collapse) |
| **PaGeR** (2026) | 360 | **single image** | n/a (monocular) | ✓ (per-room) | ~ | – | our per-room geometry backbone (not a competitor) |

**Reading:** these reconstruct **what was observed**; none completes unseen regions, and all need
covisibility we don't have. They are baselines / the motivating-failure evidence, plus our backbone.

---

## Group C — 3D reconstruction + generative completion (the "3D + inpainting" side)

| Method | Input | Scope | Overlap | Pose | 3D | Gen (unseen) | Multi-room? |
|---|---|---|---|---|---|---|---|
| **NoPoSplat** (2024) | P | scene | sparse **unposed but overlapping** | pose-free | ✓ (3DGS) | – (reconstructs observed) | – |
| **Splatter-360** (CVPR'25) | 360 | 1–2 rooms | **wide-baseline pair (shared view)** | needs poses | ✓ | ~ | – |
| **Pano2Room** (2024) | 360 | **single room** | one pano | given | ✓ | **✓ (RGB-D inpaint)** | – |
| **PanoImager** (IROS'26) | 360 | **one space** | few weak-parallax views | SfM-free | ✓ | ✓ (geom-cond. diffusion) | – |
| **CAT3D / ReconFusion / ZeroNVS** | P | object/scene | few views | given/est | ~ | ✓ (NVS diffusion) | – |
| **PanoWorld** (2026) | 360 | **whole house** | n/a | **needs floorplan input** | ~ | ✓ (**generative, not reconstructive**) | ✓ but generative |
| **→ OURS (Paper phase-2)** | 360 | **whole floor** | **near-zero, one-per-room** | **estimated** | ✓ | ✓ (through-door + unseen) | **✓ reconstructive** |

**Reading:** single-room / single-space generative completion is **DONE** (Pano2Room, PanoImager) —
so "complete one room from one pano" is **not novel**. Whole-house generation exists but is
**generative-from-a-floorplan** (PanoWorld), not reconstructive-from-panos. Multi-view 3DGS exists but
needs overlap (NoPoSplat, Splatter-360).

---

## White-space verdict

Cross the two axes that matter — **overlap regime** (x) × **output richness** (from pose → 3D → generative
completion) (y):

- **Pose/2D-layout @ near-zero overlap:** occupied (Extreme SfM; us). *Crowded, small deltas.*
- **3D reconstruction @ dense/overlap:** occupied (PanoVGGT, IM360, NoPoSplat). *Not our regime.*
- **Generative completion @ single room / one space:** occupied (Pano2Room, PanoImager). *Not novel alone.*
- **Generative whole-house @ from-floorplan:** occupied (PanoWorld). *Generative, not reconstructive.*
- **★ Reconstructive, generative, MULTI-ROOM 3D floor from ONE pano per room at near-zero overlap,
  conditioned on connectivity, under ESTIMATED poses → EMPTY.** This is the defensible cell.

The novelty is **not** any single ingredient (per-room GS, depth, view-diffusion, connectivity all
exist). It is the **combination + setting**: assembling a *consistent* multi-room floor from one pano
each, where the only cross-room constraint is the doorway, and using **connectivity as the structural
prior** that keeps rooms mutually consistent through doorways.

---

## Honest "does it already exist?" check, per candidate contribution

| Candidate contribution | Closest prior | Verdict |
|---|---|---|
| Connectivity + pose from sparse panos | Extreme SfM, SALVe, BADGR, CovisPose | **Crowded.** Don't sell as the core. Our edge = explicit connectivity graph + flip negative + zero-overlap regime. |
| Per-room 3D completion from one pano | Pano2Room, PanoImager | **Exists.** Not novel by itself. |
| Multi-room 3D floor, one-pano-each, connectivity-conditioned generative completion | — (none) | **Open.** The paper. |
| **How much pose error multi-room 3D tolerates** (our Step-0/pose-sensitivity angle) | — (unstudied for this setting) | **Open + scientific.** Strong secondary novelty; ties pose work to 3D quality. |

**Two risks this map surfaces (bring to the prof):**
1. If we outsource pose to BADGR/Extreme SfM, our connectivity/pose work becomes a *baseline*, and the
   paper must stand on the multi-room completion + consistency. Verify BADGR even runs at our overlap.
2. Step-0 showed coverage is easy under GT poses, so the completion "wow" is modest — the load-bearing
   novelty is **multi-room consistency under imperfect poses**, i.e. the pose-robustness result, with
   completion as the last mile. Frame the paper around that, not around "another NVS diffusion."

## Sources
- BADGR — https://badgr-diffusion.github.io/ · https://arxiv.org/abs/2503.19340 · CVPR'25 Highlight.
- Extreme SfM — https://github.com/aminshabani/extreme-indoor-sfm · https://openaccess.thecvf.com/content/ICCV2021/html/Shabani_Extreme_Structure_From_Motion_for_Indoor_Panoramas_Without_Visual_Overlaps_ICCV_2021_paper.html
- Others: see `docs/RelatedWork.md` (STAR survey Pintore et al. 2026 is the taxonomy anchor).

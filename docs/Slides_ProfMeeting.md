# Slide outline — supervisor update (6 slides)

*Discussion deck: what we did → what we found → where the field's white space is → the decision to
make. Each slide: what's ON it, the figure/number to show, and what to SAY. Compress to 4 by merging
2+3 and 5+6 if time is short. Numbers grounded in `ResearchLog.md`; landscape in `Landscape_Matrix.md`.*

---

## Slide 1 — Problem & where we are (anchor)

**On the slide**
- One sentence: *from ~one 360° panorama per room with near-zero overlap (except through doorways),
  recover room connectivity, metric layout, and a 3D floor.*
- Pipeline strip: per-room geometry → doors → **connectivity** → **metric pose** → **3D completion**.
- Tag the two middle boxes "done/validated", the last "in progress".

**Figure/number:** the 5-box pipeline strip (from `ProjectOverview.md`).

**Say:** The hard part is that at near-zero overlap classical SfM has no correspondences — *except at
doorways*. Our core idea: the doorway is the one shared structure carrying a cross-room constraint.
Today I'll show results across the pipeline and then a landscape map that frames one decision for us.

---

## Slide 2 — Results I: connectivity & pose (the registration side)

**On the slide**
- **Connectivity: assign-AP 0.913 (GT doors) / 0.842 (detected)**, 197 scene-disjoint ZInD homes
  (random ≈ 0.31). Global 1-to-1 assignment beats pairwise (0.737 → 0.913).
- **Which-side flip from appearance ≈ chance (0.45; 0.50 even under GT pose)** — a rigorous *negative*:
  necessitates metric anchors.
- **Camera→door distance A/B (20 homes):** DAP 0.69 m vs PaGeR 0.93 m vs depth-free fixed-width 0.61 m
  (median). → lever is a *learned per-door distance head*, not a depth swap.
- **Detector recall diagnosis:** of misses, 44% open passages (geometrically recoverable), 35%
  flush/closed (ceiling), 21% near-miss.

**Figure/number:** small table of the four results; highlight 0.913/0.842.

**Say:** Connectivity is solid and the flip-ambiguity is a clean negative result, not a failure. The
A/B is an example of the data correcting our intuition — PaGeR *looks* better but isn't better for
metric distance. Recall is the known gap, and it's ~half geometrically addressable.

---

## Slide 3 — Results II: geometry FMs & the Step-0 3D probe

**On the slide**
- **overlap_probe (our benchmark):** PanoVGGT (SOTA feed-forward panoramic) degrades sharply as
  overlap→0 — dense relRot **5.7°** → sparse **23.9°** (ATEnorm 0.105 → 0.271); within-scene strata
  1.7°/6.9°/17.2° for same/adjacent/far — **doorway-adjacent pairs excepted**.
- **Argus:** could not be reliably applied to ZInD (rig-coupled geometry) — reported as a null/transfer
  failure, not a number.
- **Step-0 3D prototype (new):** per-room GS at GT poses, render through the doorway → **disocclusion
  is low (~2–10%)**; renders are *coherent and complete*, the artifact is point sparsity, not missing
  geometry.

**Figure/number:** overlap_probe dense-vs-sparse bars; one Step-0 through-door render (0053 @0.75).

**Say:** Two messages. (1) Foundation models need overlap and collapse without it — the doorway is
where pose signal survives; that's our empirical motivation. (2) Step-0's surprise: under *correct*
poses, one pano per room already covers the floor. So coverage isn't the 3D bottleneck — **pose is.**

---

## Slide 4 — Landscape: where the white space is

**On the slide**
- The 2D map: x = overlap needed (more→less), y = output richness (pose → 3D → generative completion).
- Pose/2D corner **crowded** (Extreme SfM, SALVe, CovisPose, BADGR). 3D FMs need overlap (PanoVGGT,
  NoPoSplat, IM360). Completion exists but **single-room** (Pano2Room) or **from-floorplan** (PanoWorld).
- **Empty cell (★): reconstructive, generative, MULTI-room 3D floor from one pano per room, under
  estimated poses.**

**Figure/number:** the white-space map (SVG rendered in chat — export/screenshot it).

**Say:** This is the overview I want your read on. Everything on the pose row already exists — BADGR
even beats us on pose accuracy *where its overlap regime holds* (verify it holds in ours). The one
unoccupied intersection is multi-room reconstructive completion at our sparsity.

---

## Slide 5 — Candidate contributions: does it already exist?

**On the slide** (table)

| Candidate contribution | Closest prior | Verdict |
|---|---|---|
| Connectivity + pose from sparse panos | Extreme SfM, SALVe, BADGR | **Crowded** — not the headline |
| Per-room 3D completion from one pano | Pano2Room, PanoImager | **Exists** — not novel alone |
| Multi-room, connectivity-conditioned generative completion | — | **Open — the paper** |
| How much pose error multi-room 3D tolerates | — (unstudied here) | **Open + scientific** |

**Say:** The novelty is not any single ingredient — it's the combination + setting. And Step-0 says the
load-bearing result is **multi-room consistency under imperfect poses**, with completion as the last
mile — not "another NVS diffusion."

---

## Slide 6 — Decision & next experiment

**On the slide**
- **Decision for us:** one paper — where does the emphasis sit?
  (a) system (panos→3D floor), (b) registration-centric, (c) pose-robustness of multi-room 3D.
- **If we outsource pose (BADGR/Extreme SfM):** connectivity becomes a baseline; verify BADGR even
  runs at our overlap; the paper stands on completion + consistency.
- **Next experiment (cheap, decisive):** inject realistic pose error into a room (yaw 5–25°, trans
  0.1–0.5 m from overlap_probe) → measure how fast the through-door 3D degrades = the pose-tolerance
  curve. Reuses Step-0.

**Say:** My proposal: frame around multi-room consistency under estimated poses, with connectivity as
the enabling prior. The pose-sensitivity experiment settles whether that story has legs before we
invest. I'd like your steer on the framing.

---

### Notes for building the actual deck
- Slides 2 & 3 can merge into one "results" slide to hit 5; slides 5 & 6 can merge to hit 4.
- Reusable assets: pipeline strip (`ProjectOverview.md`), white-space map (SVG in chat), Step-0
  through-door render (`results/gs_prototype/...`), overlap_probe plot.
- Keep one number per line; let the map and one render carry the visual weight.

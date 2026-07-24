# Research Briefing — Sparse-Panorama Room Connectivity & 3D (July 2026)

*Catch-up for the supervisor meeting. Self-contained: problem → what's new since last time →
the full pipeline (high-level, then each step in detail) → outlook → landscape/positioning.
Numbers are grounded in `ResearchLog.md`; deeper detail in `ProjectOverview.md`, `Roadmap.md`,
`RelatedWork.md`.*

---

## 1. The problem in one paragraph

From a **sparse set of gravity-aligned 360° panoramas — roughly one per room, with near-zero
visual overlap except through doorways** — recover (1) the **room-connectivity graph** (which rooms
connect, via which door), (2) **metric relative poses / an SE(2) floor layout**, and (3, the thesis
goal) a navigable **3D reconstruction** of the whole floor. No poses and no floor plan are given at
test time. This is hard because classical SfM needs cross-view correspondences and, at near-zero
overlap, there are almost none — *except at doorways*. Our **core hypothesis: a doorway is a
geometric aperture** — pixels inside it are rays into the neighbouring room, so it is the one
structure carrying a cross-room constraint. Each door match still fixes pose only up to a
**which-side flip**.

---

## 2. What's new since the last meeting — and what we learned

Five threads advanced. The headline is that we now have a **working Step-0 3D prototype** and it
already changed our view of where the Paper-2 difficulty lies.

**(a) Infrastructure consolidated onto foundation models.** The repo was restructured into a
dataset-agnostic package; connectivity reproduces (0.913 GT / 0.842 detected). All foundation
models (PaGeR, PanoVGGT, Argus, VGGT) now share one `fmodels` environment. This is plumbing, but it
unblocked everything below.

**(b) PaGeR adopted as the per-room geometry backbone.** PaGeR (ETH, monocular panoramic
depth+normals) was integrated as a drop-in for our old depth (DAP) and run on 20 held-out homes.
*Qualitatively it is far better than DAP* — it resolves each doorway as a distinct depth recess and
gives crisp planar walls, exactly the through-door geometry we care about.

**(c) …but a controlled A/B corrected an over-claim.** We measured what actually matters for pose —
**camera→door distance** — across 20 homes. Result (median | MAE error vs GT):

| depth source | median err | MAE |
|---|---|---|
| PaGeR | 0.93 m | 1.38 m |
| DAP | **0.69 m** | 1.05 m |
| fixed-width geometry (no depth) | **0.61 m** | 2.48 m |

**Knowledge gained:** PaGeR's visually crisper depth does **not** translate into better metric door
distance — DAP wins on 16/20 homes, and a *depth-free* geometric estimate (door's own angular width)
beats both on the median. So the lever for metric pose is a **learned per-door distance head**, not a
depth swap. PaGeR is kept for its differential strengths (normals, opening detection, per-room 3D),
not for distance. *This is a good example of a qualitative impression being falsified by the right
quantitative test.*

**(d) Detector-recall diagnosed — the gap is ~half geometrically addressable.** The deployable
connectivity number (0.842 with detected doors) is limited by **detector recall**. Of 816 missed
doors: **21% near-miss** (fixable by cubemap-face detection), **44% open passages** (a PaGeR
depth-gap proposer could catch these — SegFormer skips leaf-less openings), **35% flush/closed** (an
appearance ceiling nothing fixes). So ~65% of misses are potentially recoverable, dominated by open
passages. A real, bounded Paper-1 win exists — **scoped but deliberately deferred** to prioritise 3D.

**(e) Step-0 3D prototype built and run (the main new result).** For a home with two connected
rooms, we place per-room Gaussians from pano+depth at **GT poses** (skipping pose/connectivity
entirely) and render a novel view *through the doorway into the neighbour room*, then measure
**disocclusion** — the fraction unobserved, i.e. what a generative prior would have to hallucinate.

- Disocclusion is **low and roughly flat** as the camera moves from the door into room B: 0053
  ≈ 0.06→0.11, 0032 ≈ 0.07–0.09, 0023 ≈ 0.02–0.03 (density-proxy metric).
- **Looking at the renders is the key insight:** the novel views are **coherent and complete** —
  each room is fully self-observed by its own 360° pano. The visible artifact is **point sparsity
  (dottiness), not missing geometry.**

**Knowledge gained (the important one):** *Under GT poses, coverage is essentially not the
bottleneck.* With one 360° pano per room, each room sees itself; the residual a completion prior must
fill (grazing angles, furniture, doorjamb) is small. The dottiness is a **rendering** limitation that
real Gaussian surfels fix — engineering, not science. **This redirects Paper 2: the determinant of
reconstruction quality is POSE/alignment, not view-completion.** (Consistent with overlap_probe: the
sparse-view pose regime is hard — 23.9° rotation, 0.271 normalised ATE.)

*In progress / blocked on cluster:* a **gsplat surfel** render to confirm the true alpha-hole number
(the CPU metric is a proxy). First GPU run succeeded end-to-end after threading two cluster library
shadows; a splat-size fix is queued (splats were sized for full-res and left grid-gaps at stride 4).

---

## 3. The pipeline — high level, then each step in detail

```
[1] per-room geometry  →  [2] doors  →  [3] connectivity graph  →  [4] metric SE(2) layout  →  [5] 3D completion
        (FM: PaGeR)         (detect +        (OUR contribution,        (door-anchored pose        (Paper 2:
                             embed)          Paper 1 core)              graph, flip resolved)      GS + diffusion)
```

**Step 1 — Per-room metric geometry.** *Input:* one panorama per room. *Method:* a monocular
panoramic foundation model (PaGeR) gives metric depth + surface normals per room. *Status:* solved by
off-the-shelf FMs — **not our contribution**; we consume it. *Open:* which depth to trust for which
job (A/B above: DAP for distance, PaGeR for normals/geometry).

**Step 2 — Door detection + cross-view door embedding.** *Input:* the panoramas. *Method:* detect
doors (SegFormer on ring views), then embed each door with a learned descriptor (DINOv2 + head,
symmetric InfoNCE, same-home hard negatives) so the *same physical door seen from two rooms* embeds
similarly. *Status:* embedding works; detection is the weak link. *Open:* **recall** (Step-2d above) —
cubemap-face detection + PaGeR geometric opening proposals.

**Step 3 — Connectivity graph (Paper 1 core, our contribution).** *Input:* door embeddings across all
rooms. *Method:* pose the cross-room door matching as **global 1-to-1 assignment** (not independent
pair scoring) and read off which rooms connect via which door; report a threshold-free **assign-AP**.
*Status:* **DONE — 0.913 with GT doors, 0.842 with detected**, on 197 scene-disjoint held-out ZInD
homes (random ≈ 0.31). Global assignment beats pairwise (0.737→0.913 with hard negatives + backbone
unfreeze). *Open:* close the 0.913→0.842 gap via detector recall; add a high-precision operating point.

**Step 4 — Metric SE(2) layout (door-anchored pose graph).** *Input:* the connectivity graph +
per-room geometry. *Method:* each matched door gives a relative pose (up to the which-side flip);
match confidence weights a robust SE(2) pose graph; optional flip-free SfM anchors pin scale/flip.
*Status/knowledge:* the **which-side flip from appearance is a genuine geometric ambiguity** — flip
accuracy ≈ chance (0.45; 0.50 even under GT-oracle pose), and real floors are trees so
cycle-consistency can't fix it either. This is reported as a **rigorous negative result** that
*necessitates* metric anchors. A hybrid pose graph with 15–30% flip-free anchors cuts layout error
~6× (mechanism shown; needs a real COLMAP/SfM run). *Open:* the SfM anchor (C5), ablation hygiene.

**Step 5 — 3D completion (Paper 2, the forward track).** *Input:* per-room geometry + connectivity +
poses. *Method:* per-room 3DGS (holey where unobserved) → a **diffusion prior conditioned on the
partial render + connectivity/doorway constraints** hallucinates unobserved regions and through-door
views → distilled back into GS (NoPoSplat/CAT3D-style) → refine → navigable floor. *Status:* **Step-0
prototype done** (this month). *New knowledge from Step-0:* under GT poses coverage is largely solved,
so the live question shifts to **how much pose error the reconstruction tolerates** before it breaks.

---

## 4. Outlook — what's planned

**Immediate next experiment (cheap, high-value, reuses Step-0).** **Pose-sensitivity test:** inject
realistic pose error into room B (the overlap_probe range: yaw ~5–25°, translation ~0.1–0.5 m) and
measure how fast the through-door reconstruction degrades. This directly tests the hypothesis that
**pose, not completion, is the Paper-2 bottleneck**, and it needs only a proper misalignment metric
(masked PSNR of the novel view at GT vs perturbed pose — the density/hole metric won't see ghosting).

**Confirm the Step-0 finding with real surfels.** Finish the gsplat run (queued splat-size fix) so the
"coverage is solved under GT poses" claim rests on a true alpha-hole map, not the CPU proxy.

**Then Paper 2 staged (kill-criteria, do not build everything at once):**
1. GT poses on 2–3 rooms → per-room GS → off-the-shelf view-diffusion completes one through-door
   transition. *Kill if diffusion can't plausibly complete a single through-door view.*
2. Add connectivity conditioning; test cross-room consistency vs unconditioned.
3. Scale to the whole floor; evaluate novel-view PSNR/SSIM/LPIPS at held-out doorway views + walkthroughs.

**Paper 1 loose ends (bounded, in parallel):** detector recall (cubemap + PaGeR opening proposals,
the ~44%-open lever); the real SfM/COLMAP anchor for C5; ablation-grid hygiene; a learned per-door
distance head (beat the 0.61 m fixed-width median without its wide-door MAE blow-up).

**Data:** add Stanford2D3D to the dataset abstraction for a direct depth metric and an in-distribution
overlap_probe run (addresses the ZInD-OOD caveat).

---

## 5. Landscape & positioning — what exists, where we sit

**The unifying insight:** *every* strong 2026 method needs cross-view overlap; **none targets our
near-zero-overlap, one-pano-per-room regime.** They differ only in how they exploit overlap. Our lane
is **registration + completion where covisibility collapses to the doorway.** The 2026 STAR survey
(Pintore et al., Eurographics) is our anchor and names the subproblem "positioning vs connecting rooms."

**Foundation models — solve the pieces around us, need overlap:**
- **PaGeR** (ETH/Google 2026) — monocular panoramic depth+normals+metric via a cubemap lift of Depth-
  Anything-3. SOTA single-image geometry. *Not a competitor — our per-room backbone.* Does no
  multi-room registration.
- **PanoVGGT** (CVPR'26) — panoramic VGGT; our probe shows it collapses as overlap→0, doorway-excepted.
  *This is the empirical evidence for our motivating claim.*
- **Argus** (ECCV'26), **IM360** (ICCV'25) — feed-forward / SfM panoramic 3D; both need substantial
  covisibility (dense-capture competitors, not near-zero-overlap solutions).

**Closest neighbours — read these carefully for the meeting:**
- **NoPoSplat** (Microsoft) — **pose-free feed-forward 3DGS from sparse *unposed* images**, canonical-
  frame anchor, intrinsics-as-token for scale. *This is the closest to "what we want" — but it assumes
  overlapping views.* It is our **Paper-2 architecture blueprint**; the novelty is pushing it to zero
  overlap via the door/connectivity structure and adding generative completion.
- **BADGR** (CVPR'25 Highlight) — **diffusion-based bundle adjustment** jointly denoising poses + 1D
  layouts from wide-baseline sparse panos (as low as 0.6 images/room), same ZInD lineage. *Likely our
  closest 2025 pose-side competitor.* Apparent difference: it does **not** output an explicit
  connectivity graph (to verify by full read) — connectivity-as-first-class-output may be our edge.

**Direct sparse-multi-room lineage (Paper-1 baselines):**
- **Extreme SfM** (ICCV'21) — origin: sparse panos, no overlap, single-image layout + geometric
  optimisation → coarse floor plan.
- **SALVe** (ECCV'22) — W/D/O cues hypothesise pairwise alignments, CNN verifies from BEV, pose graph →
  2D plan. *Our primary 2D baseline.* We add threshold-free connectivity AP, the flip-necessity
  analysis, and a layout-free door embedding.
- **NadirFloorNet / PanoFloor** (2025) — build the floor-plan/connectivity graph but **assume room
  positions are given**; we solve their upstream input (connectivity *and* pose from pixels alone).

**Completion / GS (Paper-2 technique family):**
- **Splatter-360** (CVPR'25), **PanoImager** (IROS'26), **Pano2Room** — panoramic/near-single-room GS +
  inpainting; single- or paired-room, still need shared view. Technique reservoir for our completion
  stage, not multi-room solutions.
- **PanoWorld** (2026) — generative whole-house panorama synthesis from a floor plan; *generative not
  reconstructive*, but its cross-room consistency mechanism is worth borrowing.

**One-line positioning for the slide:** *Foundation models now solve panoramic geometry (PaGeR) and
dense/overlapping reconstruction (PanoVGGT, IM360, Argus, NoPoSplat). The open problem — our lane — is
registration + generative completion of a multi-room floor from one pano per room, where covisibility
collapses to the doorway.*

---

## 6. Talking points for the meeting

1. Connectivity is solid (0.913/0.842, 197 homes) and the flip-ambiguity negative is a *clean,
   defensible result*, not a failure.
2. The A/B and the Step-0 prototype show we are **letting the data correct our intuitions** (PaGeR
   isn't better for distance; coverage isn't the 3D bottleneck).
3. Step-0's message: **under GT poses, one-pano-per-room already covers the floor** — so Paper 2's
   real question is *pose tolerance*, and connectivity/layout (Paper 1) feeds directly into it.
4. We are close to NoPoSplat and BADGR but distinguished by the **regime (zero overlap, one per room)**
   and by **connectivity as an explicit, first-class output** conditioning the completion.

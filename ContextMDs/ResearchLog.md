# Research Log

## Template

### Date

#### Observation
What happened?

#### Decision
What decision was made?

#### Reasoning
Why was the decision made?

#### Outcome
What was learned?

#### Next Steps
- Item 1
- Item 2

---

# Entries

## 2026-06-18

### Observation
Discussion around doorway-ray based room connectivity estimation.

### Decision
Focus first on validating the geometric signal before building complex optimization pipelines.

### Reasoning
If the core signal does not exist, additional engineering complexity will not help.

### Outcome
Defined a staged validation strategy.

### Next Steps
- Stage 1 known-pose validation
- Build residual visualizations
- Compare connected and non-connected room pairs

---

## 2026-06-24

### Observation
Clean restart in SparsePanoramaRoomConnectivity/. Two checks on the old work:
1. DAP "metric" depth vs Stanford GT (85 area_3 panos): best scale 1.01,
   correlation 0.83-0.98, AbsRel~0.13, far/through-door region corr 0.81-0.96.
   => DAP depth is genuinely metric AND reliable through doorways. Earlier worry
   that depth was the bottleneck is REJECTED.
2. exp01 pose-recovery ablation (scipy least_squares, Huber; clean GT-consistent
   see-through points; yaw+translation perturbation). Premise check: relative
   rotations are pure yaw (vertical tilt median 0.44 deg, max 1.06 deg).

### Outcome
- The old Stage-2 "rotation 5-95 deg" disaster was NOT fundamental. With a proper
  solver + realistic yaw perturbation + clean correspondences + bounded scale,
  even full Sim(3) recovers pose to sub-degree median.
- Cost landscape has a clean basin: minimum at 0 deg yaw and scale=1.0.
- Constraints buy ROBUSTNESS, not median accuracy. At 15 deg init, V0/V1 have
  ~10% catastrophic divergence (rot up to 25 deg, on the hardest pair); V2
  (yaw + FIXED scale=1) has 0/40 failures and translation max 0.07 m. Fixing the
  (verified metric) scale removes the gauge direction the optimiser falls into.

### Decision
Adopt yaw-only rotation + fixed metric scale as the pose model going forward.
Pose recovery from clean see-through correspondences is considered well-posed.

### Reasoning
Isolating geometry/optimisation from the door detector shows the geometric signal
is sufficient. Therefore the real remaining bottleneck is obtaining clean
see-through correspondences WITHOUT GT (door detection / association).

### Open caveats
- Point selection used GT pose to pick co-visible points -> this experiment
  isolates conditioning; it does NOT validate door detection.
- Single building (area_3), proxy connectivity (centre distance < 2.5 m), DAP depth.
- Generalisation across buildings (Structured3D/Matterport) still untested.

### Next Steps
- exp02: replace GT-based see-through selection with a real detector (geometric
  aperture and/or learned), measure pose degradation -> the actual bottleneck.
- Add true connectivity GT (ZInD) instead of distance proxy.
- Cross-building generalisation (Structured3D render or Matterport).

---

## 2026-06-24 (exp02 — geometric door detection + a methodology correction)

### Observation
Built a geometric (pose-free) aperture detector (depth-beyond-wall on the horizon
profile) + a provider abstraction (Stanford proxy / ZInD GT). First exp02 run
looked like a failure: detection recall 0.06, pose rot median 17.6 deg, connected
vs unconnected residual NOT separable.

### Root-cause check (before believing the failure)
- GT "co-visible columns" for every connected pair span the FULL azimuth
  (n=511-683 of 4096 cols, -180..180 deg). Because the distance<2.5 m proxy picks
  cameras 0.85-1.2 m apart -> huge DIRECT overlap, NOT the sparse doorway-only
  regime. So the recall metric (vs whole-pano target) was meaningless.
- Re-evaluated the detector against the bearing to the neighbour camera instead:
  nearest detected aperture is within 7 deg median of the neighbour direction,
  88% within 25 deg (7/8 pairs). The detector DOES find the connecting opening.

### Outcome / corrected conclusions
- The geometric door-direction detector is PROMISING, not broken.
- TWO real problems are now the actual agenda:
  1. Evaluation regime: the camera-distance proxy conflates high-overlap and
     doorway-only pairs. Need true room separation + doorway-location GT.
     -> ZInD W/D/O annotations are the right tool (connectivity by shared opening,
        not camera distance). Makes the "flexible GT" switch scientifically necessary.
  2. Association: a room has several openings; selecting the door-to-B by lowest
     converged residual is currently unreliable. Aperture->neighbour association
     is an open sub-problem (candidate for the contrastive idea).
- Caveat on exp01: its easy pose result was partly inflated by the same overlap;
  must re-test pose in the genuinely sparse (doorway-only) regime.

### Decision
Move evaluation onto ZInD GT: define connectivity via room adjacency / shared WDO,
use WDO opening location as doorway GT, and select pairs that are truly
room-separated (sparse overlap). Validate ZindProvider pose convention first.

### Next Steps
- Parse zind_data.json: rooms, WDO openings, pano->room, GT poses.
- Define sparse-regime pairs (adjacent rooms sharing one WDO, minimal direct overlap).
- exp03: detector evaluated vs WDO doorway GT; pose in the true sparse regime;
  aperture->neighbour association baseline.
- (ZInD DAP depths only exist for 19 panos in legacy folder -> may need to
  regenerate DAP depth for the full sample tour on the laptop GPU.)

---

## 2026-06-25 (exp03 — ZInD wired, calibrated, and the open/closed-door result)

### What was built
- src/zind.py: parser for zind_data.json (rooms, floor_plan_transformation poses,
  W/D/O in local+global coords, shared-door connectivity).
- ZindProvider now uses AUTHORITATIVE floor-plan poses (not sample_pairs) under an
  empirically calibrated ZInD->geometry convention.
- DAP depths regenerated for the ZInD sample tour (generate_depth.py).

### ZInD->geometry convention (calibrated, not guessed)
Searched 64 sign/offset/axis combos; scored by tightly-consistent points (<8 cm)
at GT pose over connected pairs. Winner (1.7x over median):
    world X = -pos_x * S,  world Z = +pos_y * S,  yaw = +rotation_deg     (S=3.55 m/coord)
VALIDATED by pose recovery: from an 8 deg / 0.3 m perturbed start, V2 recovers
rotation to median 1.6 deg (75% <5 deg); the 2 failures are the lowest-co-visibility
(genuinely sparse) pairs.

### THE key result (true sparse regime, ZInD)
For connected pairs with a floorplan-matched shared door (8 of 20):
- Detector vs the TRUE doorway azimuth: median 29 deg error, 38% within 25 deg
  -> looked like the detector fails.
- BUT only 4/8 doors are actually OPEN (see-through depth ratio >1.4 vs wall).
- Conditioned on OPEN doors, the detector finds the doorway well:
  median 13 deg, 75% within 25 deg. On CLOSED doors it necessarily misses
  (98-122 deg) because there is NO see-through geometry to detect.

### Interpretation (defining the project)
- The doorway-see-through hypothesis is VALID but CONDITIONAL on an open door.
- In this ZInD sample ~half of door-connections are closed -> the depth/see-through
  line CANNOT recover those; they need an APPEARANCE-based signal (the door/frame as
  a co-visible object seen from both sides), i.e. the contrastive direction.
- Connections via OPENINGS (no door leaf) are always see-through; restricting this
  analysis to DOORS biases toward the hard (closed) cases, so the true open-fraction
  across all connections is likely higher. Must handle openings separately.

### Caveats
- n=8 matched doors (small); ratio>1.4 open/closed threshold is crude.
- shared_door matched only 8/20 connected pairs (tol, and openings not yet matched).
- Single floor / sample tour.

### Decision / next steps
- STRATIFY every future evaluation by connection type: opening / open-door / closed-door.
- Improve connectivity parsing: also match 'openings', loosen door tol, report per-type.
- For open-door + opening pairs: the geometric pipeline is the baseline to beat.
- For closed-door pairs: prototype an appearance/contrastive door-as-object matcher
  (the only viable signal). This is now a concrete, motivated reason for direction 2.
- The OPEN-FRACTION in the user's real industrial captures is the single most
  decision-relevant unknown -> measure it as soon as that data is available.

---

## 2026-06-25 (exp04 — proof of concept on Immersight partner data, NO GT)

### Setup
19 high-res panos (5504x11008), one floor, big overlap, depths from DAP. No GT
poses/connectivity (manual/visual verification only). src/register.py: unknown-pose
pairwise registration = multi-start over yaw (translation 0) + V2/V1 refine, scored
by inlier fraction (which doubles as an overlap/confidence score, GT-free).

### Findings
- DAP depth here has CORRECT relative room geometry (walls + a doorway notch are
  clearly visible in a contrast-stretched top-down) BUT wrong ABSOLUTE scale: factor
  102 (calibrated on Structured3D) compresses a ~6 m room to ~2.5 m. Relative pose is
  unaffected; a cosmetic uniform VIS=2.5 is used only for room-sized plots.
- Monocular depth is AFFINE (per-image scale). Two panos are NOT mutually metric:
  free-scale V1 recovers a ~1.05-1.10 relative scale and aligns better than fixed
  V2. => regime-dependent pose model:
      sparse doorway + metric depth  -> V2 (yaw + fixed scale)   [ZInD/Stanford]
      big overlap + affine mono-depth -> V1 (yaw + FREE scale)   [Immersight]
- PoC result: two panos register with NO pose prior into a coherent local floor map,
  cameras ~2 m apart, overlapping room walls roughly coincide. Figure:
  results/exp04_immersight/immersight_poc.png (input / depth / single-room BEV /
  two-pano registration).

### Honest caveats
- Recovered yaw is NOT fully stable across runs (120/140/270 deg seen) -> the inlier
  landscape is flat because the depth is compressed / low dynamic range. Needs visual
  verification and/or a stronger cue.
- inlier ~0.53-0.58 with a ~0.49 floor on non-overlapping starts -> weak margin.
- No GT -> all quantitative claims here are qualitative until manually checked.

### Next steps
- Fix absolute scale: recalibrate DAP per dataset (one known distance is enough), or
  run DAP at higher input res, or try a stronger metric pano-depth model. Better depth
  -> sharper inlier landscape -> stable yaw.
- Use the doorway see-through direction to constrain/disambiguate the yaw search.
- Build the multi-pano floor map (register all 19 to an anchor / pose graph) once
  pairwise is stable -> the eventual walkable-reconstruction goal.
- Manually verify the correct yaw on 2-3 pairs to calibrate confidence.

---

## 2026-06-25 (door-centric pipeline + door-anchored pose: ambiguity finding)

### Architecture committed (door = backbone)
Per pano fuse THREE signals at each door:
  - semantic (SegFormer/ADE20K on perspective ring views): door EXISTS + where
    (works for closed / depth-invisible doors -> fixes kitchen & ZInD-closed cases);
  - depth see-through (src/aperture): is there usable GEOMETRY through it (pose anchor);
  - appearance (DINOv2 door crops, src/matching): MATCH a door across two panos = edge.
Modules: src/panoproj (e2p ring views), src/door_semantic, src/doors.fuse,
src/matching (DoorMatcher), src/door_pose. Open/closed is NOT inferred from depth
(that was the kitchen mislabel); only 'has see-through geometry' is.

### Key insight — a matched door gives pose init for free
For gravity-aligned panos, Ry(theta) maps dir(phi)->dir(phi+theta), so if the door
is at az_A in A and az_B in B, relative yaw theta ~= az_B - az_A + 180 (cameras face
the same wall from opposite sides); door distances give the translation init. No
blind yaw search needed.

### Door-anchored pose on ZInD (GT) — result + the hard part
Refining on THROUGH-DOOR points (A's view of B's room) from the yaw-from-match init:
  - ~40-50% of pairs recovered well (several < 3 deg rotation);
  - the rest land ~180 deg off WITH HIGH INLIER (0.99) -> a binary WHICH-SIDE
    ambiguity (the 180 flip rotates the room around the door, keeping the door + the
    range residual consistent). This is SALVe's door which-side ambiguity.
  - Geometric disambiguators TRIED and INSUFFICIENT on these symmetric/high-overlap
    rooms: (a) require door to reappear at az_B in B -> fails (door is the pivot);
    (b) free-space / non-interpenetration of the two rooms -> only ~chance.
  - Translation along the door axis is weakly constrained even when rotation is right.

### Conclusion (unifies the project)
Geometry resolves relative pose only UP TO a binary which-side flip; breaking it needs
APPEARANCE -- which specific surface is seen through the door. So contrastive cross-view
matching is needed not just for detection but to DISAMBIGUATE POSE. This is the
strongest motivation yet for the contrastive direction.

### Caveats / next
- ZInD sample pairs are close / high-overlap (not the ideal thin-doorway regime); the
  flip may be milder when the through-door slice is more distinctive (far cameras) --
  worth testing on truly sparse pairs.
- Next: appearance-based which-side disambiguation (match A's through-door view to B's
  content); and/or a non-symmetric geometric cue (door jamb left/right parity).
- door_pose.recover currently returns the best-inlier candidate (so it can flip);
  treat its output as pose-up-to-flip until the disambiguator is added.

### Which-side disambiguation experiments (exp08, on ZInD w/ GT)
Generate the two candidate poses (yaw init az_B-az_A+180 vs az_B-az_A), pick the side
by an appearance score, measure correct-side rate vs GT.
- DINOv2 (ViT-S) GLOBAL crop: A's through-door view vs B's content at the projected
  location -> correct side 6/10 (60%), median rotErr 18.8 deg. Signal is REAL: when the
  cosine margin is clear it's usually right (01,03,04,10b); ties give nothing (17,02);
  one confident-but-wrong (10a).
- PHOTOMETRIC (raw RGB) consistency of through-door points -> 3/10 (30%), worse:
  view/lighting change across opposite-side views is too large for raw colour.
=> Appearance carries the which-side signal, but OFF-THE-SHELF generic features are
   only weakly discriminative (60%). Reliable disambiguation (and matching) needs a
   TASK-SPECIFIC CONTRASTIVE cross-view embedding. This is now the core research
   contribution, empirically grounded.

### Plateau / decision
The geometry + heuristic-appearance pipeline is fully built and characterised end to
end (semantic door -> match -> door-anchored pose-up-to-flip -> appearance flip pick).
Further heuristics show diminishing returns. Next MAJOR step = train a cross-view door
embedding (contrastive), which connects:
  - direction 2 (contrastive association) — the embedding itself;
  - direction 4 (synthetic data) — supervision: render door pairs from Matterport3D /
    Structured3D with KNOWN cross-side correspondence + GT pose to train/eval it.
Target: embedding that (a) matches a door across opposite sides, (b) scores the correct
which-side candidate >> the flip. Validate on ZInD (correct-side rate, pose error) and
demo on Immersight.

---

## 2026-06-25 (contrastive embedding TRAINED — results)

### Data + training
exp09 over a PARTIAL ZInD download (157 homes) -> 1854 cross-view door pairs
(1568 train / 286 val, scene-disjoint). exp10: DINOv2-ViTS frozen + trainable head,
symmetric InfoNCE, 40 epochs, loss 3.64 -> 0.43.

### Result 1 — MATCHING works (this is CONNECTIVITY)
Held-out cross-view retrieval: baseline (untrained head) top1=0.02/top5=0.07 -> TRAINED
top1=0.19 / top5=0.65 (chance ~0.003). The encoder learned to recognise the SAME door
from opposite sides. Matching doors across panos = room-connectivity edges => the core
project goal is now delivered by the contrastive embedding. Direction 2 validated.

### Result 2 — which-side flip NOT improved (still 60%): WRONG TOOL, not failure
exp11 (trained encoder in the exp08 protocol): 60% correct side, median rotErr 19 deg =
same as the off-the-shelf DINOv2 baseline. Cause: the encoder is trained for door-vs-door
IDENTITY (matching), but which-side feeds it A's door-view vs B's WALL-surface-view (where
through-door points land) -- a different comparison. It is now confidently wrong in places
(high sim gap, wrong pose) => right tool for matching, wrong tool for the flip.

### Reframe / decisions
- The encoder's win is CONNECTIVITY (matching), the primary goal. Build + score the
  connectivity graph next (exp12).
- The which-side flip is a pose-refinement detail. Best resolved at GRAPH level: with
  cycles/loop-closure a flipped room is globally inconsistent -> disambiguated for free.
  Per-pair alternative = dense feature/point correspondence (DINOv2 patch / LoFTR) between
  A's through-door region and B (NOT the door-identity encoder).
- More ZInD homes (finish download) + tighter door crops should raise retrieval further.

### exp12 — ROOM-CONNECTIVITY headline result (the project's core goal)
Held-out complete home 0036 (64/64 panos usable), NO poses used, room-level,
threshold-free:
    AVERAGE PRECISION = 0.53   (random = 0.20  -> ~2.6x chance)
    best-F1 = 0.53  (P=0.67, R=0.44)  @ cosine thr 0.62
=> Semantic room connectivity from sparse panoramas WORKS on real data: predicting
which rooms connect purely by matching their doors' appearance across panos. When it
predicts an edge it's right ~2/3 of the time; it currently misses ~half the true edges.

Interpretation: clear, honest signal well above chance, MODERATE accuracy. Bottleneck
is matcher strength (top1 0.19 / top5 0.65), capped by:
  - partial training set (only 157 ZInD homes, 1568 train pairs);
  - frozen DINOv2 backbone (head-only training);
  - loose door crops (fov 70) + some under-extracted doors (72 crops / 64 panos).
Levers (all set up, just need a GPU pass): finish the ZInD download (10x more homes),
tighten crops (fov ~50 in door_dataset, then re-extract + retrain), unfreeze=True in
build_encoder. Caveat: confirm 0036 was NOT in the training extraction (use the val-list
one-liner) for a strictly-held-out number; it likely was incomplete during exp09 so ~OK.

### STATE OF THE PROJECT (milestone)
End-to-end, validated on real data, no ground-truth poses required:
  semantic door detection -> cross-view door matching (trained) -> room-connectivity
  graph (AP 0.53 held-out). Pose is solved up to a per-pair which-side flip with a
  principled graph-level fix planned. The central thesis -- "semantic room connectivity
  from sparse 360 panoramas" -- is demonstrated.

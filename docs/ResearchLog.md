# Research Log (distilled)

*Condensed record of validated findings and decisions, newest last. Debugging blow-by-blow and
duplicated narrative have been trimmed; this is the evidence trail. One-line entries with the number
and the conclusion.*

## 2026-06 — Foundation of the approach
- **Geometric door detection + methodology correction:** always root-cause a "failure" before believing
  it. Early detectors validated on Stanford2D3D.
- **ZInD wired + calibrated** (ZInD→metric convention, gravity-aligned yaw-only). Key sparse-regime
  result: near-zero overlap except doorways → classical correspondence is absent; the doorway is the
  shared structure. Defines the project.
- **Immersight partner data (no GT):** qualitative proof of concept on real captures.
- **Door = backbone decision:** a matched door gives a relative-pose init for free (yaw from bearings,
  translation from depth) — up to the **which-side flip**. This unifies connectivity + pose.
- **Contrastive door embedding TRAINED** (DINOv2 ViT-S + head, symmetric InfoNCE): matching works →
  this IS connectivity. Which-side flip NOT improved by appearance (~60%) — flagged as wrong tool.
- **exp12 connectivity headline:** room-connectivity graph from door matching, threshold-free AP.

## 2026-06/07 — Connectivity result matured
- **Scoring:** max → mutual-NN → **global 1-to-1 door assignment** (assign) is best (a door connects to
  ≤1 door in one other room).
- **Same-home hard negatives** (batches from few homes → in-batch negatives are same-building doors) fix
  large-home precision (0.55→0.86).
- **2×2 ablation (assign AP, 197 held-out homes):** frozen-nohn 0.737 | frozen-hn 0.754 | unfrozen-nohn
  0.863 | unfrozen-hn **0.913**. → Backbone fine-tuning is the dominant lever (+0.13); hard negatives add
  +0.05 **only when unfrozen** (synergy). Corrected earlier over-attribution to hard negatives alone.
- **GT-free connectivity (exp28):** GT doors **0.913** vs detected doors **0.842** (192–197 homes).
  Detector P/R @15° = 0.31/0.66; the gap is driven by **recall** (missed openings/closed doors), robust
  to over-detection. This is the deployable number and the main remaining lever.

## 2026-07-14 — Which-side flip: rigorous negative
- **exp29 (31 held-out floors):** embedding flip accuracy **0.45** (≈ chance; geometry 0.55, free-space
  0.36). **exp33 diagnostic:** 0.48 with estimated pose and **0.50 under GT-oracle pose** → NOT depth-
  limited; the appearance signal simply does not carry the side. Cosine-margin confidence uninformative.
  The earlier 5/6, 7/7 anecdotes were survivorship on favourable floors. **Drop the flip claim; report
  as a genuine geometric ambiguity that necessitates metric anchors.**
- Real floors are trees (≈0 cycles on ~30/31 floors) → cycle consistency can't fix flips either.
- **M2 distance baseline (exp31):** camera→door distance error — DAP depth median 0.65 m; fixed-width
  geometry 0.44 m median but 1.66 m MAE on wide openings → motivates a learned per-door distance head.

## 2026-07-16 — Foundation-model probe (overlap_probe): THE result for the direction
- Ran PanoVGGT (CVPR'26 feed-forward panoramic) on 20 held-out ZInD homes, dense (all panos) vs sparse
  (one-pano-per-room). Checkpoint loads fully (valid). Gauge-robust metrics (Sim3-aligned ATE +
  alignment-free relative rotation).
- **Scene medians:** dense ATEnorm 0.105 / relRot 5.7° ; sparse 0.271 / 23.9° (rotation ~4× worse).
- **Within-scene overlap strata (rules out frame-count confound — same inference):** relRot rises
  monotonically 1.7° (same-room) → 6.9° (doorway-adjacent) → 17.2° (non-overlapping). Doorway-adjacent
  pairs stay accurate even in the sparse regime (3.2°); far pairs collapse (49°).
- **Conclusion:** feed-forward panoramic reconstruction does NOT solve the one-pano-per-room regime;
  degradation is overlap-driven; **the doorway is exactly where residual pose signal survives** — which
  argues *for* the door-based approach. Caveat: ZInD is out-of-distribution for PanoVGGT, so read the
  *relative* dense-vs-sparse and the within-scene strata (both control for model quality), not absolutes.
- **Argus (ECCV'26): INCONCLUSIVE, not reported.** Matched every documented inference step (demo
  preprocessing 196×560 crop, ref_idx reorder inverted, w2c→c2w) yet poses stay ~random on ZInD
  (dense 92°). Same harness scores PanoVGGT 5.7° and oracle 0°, so the eval is sound. Most likely Argus
  doesn't transfer to ZInD's full-sphere ERPs (coupled to Realsee's capture rig). Can't cleanly separate
  "fails to transfer" from "a last integration subtlety" → banked, not reported. Wrapper kept at
  `overlap_probe/model_wrappers/argus_infer.py`.

## Assets
- Weights: `best.pt`, `best_hardneg.pt`, `door_encoder.pt`. Frozen test split: `runs/hardneg/val_homes.txt`
  (197 homes). Clean model benchmark: `overlap_probe/`. Distance dataset substrate: `data_floors/`.


## 2026-07-2x — Infra hardening + PaGeR geometry backbone
- **Repo restructured** (dataset-agnostic): `sparsepano/` (datasets/geometry/doors/pose/gs/metrics/viz) +
  `pipelines/` (CLIs) + `benchmarks/overlap_probe/` + `docs/` + `legacy/` + `weights/`. Old `src/`,
  `experiments/`, root `config.py` removed (migrated; git history kept). Connectivity reproduces via
  `python -m pipelines.run --stage connectivity` — AP ≈0.91 GT / ≈0.84 detected (local subset 0.956).
  Cluster scripts on `python -m pipelines.*`; `env_*.sh` export PYTHONPATH; all SLURM logs → `logs/`.
- **Single shared `fmodels` venv** (torch 2.5.0+cu124) for PanoVGGT/Argus/VGGT/PaGeR (was one venv each);
  harness picks it via the per-model `*_PY` vars. Rebuild recipe frozen in `$WORK/envs/fmodels.lock`.
- **overlap_probe viz added:** 2D top-down GT-vs-pred camera overlay (`--viz`) + optional 3D point-cloud
  dump for named scenes (`--dump_ply`, GT=red/pred=green cameras). Used to *see* PanoVGGT's sparse
  collapse (0053 clean vs 0149 warped). Clean PanoVGGT result reproduced (dense 5.7° / sparse 23.9°).
- **PaGeR (ETH, monocular panoramic geometry) integrated** as a DAP-drop-in: ZInD dataloader +
  `scripts/trackF_pager_leonardo.slurm` + `scripts/pager/pager_to_pipeline.py` → per-pano metric depth +
  normals under `<home>/pager_depth/{depth_meters,normals}/`. Run forced `--scene_mode indoor` on the 20
  held-out homes. **Qualitative finding (consistent across ALL inspected samples): PaGeR >> DAP** — PaGeR
  resolves each doorway/opening as a distinct depth recess and gives crisp planar surfaces; DAP is a
  smooth low-frequency blob that smears openings into a wall. This is exactly the through-door geometry
  our door-anchored pose needs. ADOPTION still pending the quantitative `distance_baseline` scale check
  (ZInD has no GT depth → the metric decision needs the depth→door-distance vs GT test).
- **Hypothesis to test:** PaGeR's real doorway geometry may revive the *free-space* which-side flip cue
  that was ~chance with DAP (distinct from the settled *appearance*-flip negative).


## 2026-07-2x — PaGeR vs DAP A/B (camera→door distance): DAP wins — do NOT adopt PaGeR for distance
`pipelines.depth_ab`, 1424 paired doors over 20 held-out homes, camera→door distance error vs GT door
geometry (median | MAE):
  PaGeR depth        0.93 | 1.38 m     (DAP better on 16/20 homes)
  DAP depth          0.69 | 1.05 m
  fixed-width (0.9m) 0.61 | 2.48 m     (best median, worst MAE — blows up on wide/double doors)
  GT distance median 1.67 m.
=> **Counterintuitive but clear: PaGeR's crisper depth (median 1.56 m, max 5.86 m per pano) does NOT
   translate into better metric camera→door distance — DAP is better.** The qualitative "PaGeR >> DAP"
   (resolves doorways) was misleading for THIS metric. Likely causes: (a) PaGeR compresses/under-shoots
   far depth (per-pano max ~6 m — hallway doors several m away get squashed); (b) its sharp door edges
   make the bearing-sector sample lock onto the near frame/leaf rather than the door plane, so sharpness
   can hurt this particular estimator. Methodological win: the number overturned the eyeball.
Decisions:
- **Do NOT adopt PaGeR as the distance/pose depth source.** Keep DAP there for now.
- **Fixed-width geometry beats BOTH depths on the median** → for camera→door distance, the door's own
  geometry (→ a LEARNED per-door width/distance head, exp32/M2) is the right lever, not any monocular
  depth. This reinforces the PaperV2 "drop monocular depth, get distance from the door" thesis. Elevate
  the learned per-door head over any depth-backbone swap for the distance signal.
- **Keep PaGeR for its DIFFERENTIAL strengths** (this A/B doesn't test them): surface NORMALS + crisp
  per-room geometry for (i) opening/door detection → connectivity recall, (ii) the free-space which-side
  cue retest, (iii) Paper-2 per-room 3D. PaGeR's value is geometry/normals, not metric door distance.


## 2026-07-2x — Detector-recall diagnosis: the gap is ~half geometrically addressable
`pipelines.detect_diagnose`, 1560 GT doors over the 20 held-out cyclic homes, tol 15°. Per-pano-per-door
recall 0.48 (harder cyclic subset + per-pano; vs 0.66 on the 197-home set). Of the 816 MISSED doors:
  near-miss (<30°)   21%  -> cubemap-face detection (localization)
  open (depth gap)   44%  -> PaGeR GEOMETRIC opening proposal can catch (leaf-less passages SegFormer skips)
  flush (closed)     35%  -> appearance ceiling; neither lever fixes
=> The recall gap is NOT mostly irreducible: ~65% of misses are potentially recoverable, dominated by
   OPEN passages that a PaGeR depth-gap proposer could add. Real bounded Paper-1 win exists.
CAVEATS: 44% is an UPPER BOUND on recoverable recall — a depth-gap proposer also fires on windows/mirrors,
so the PRECISION cost is untested and could offset the gain; and this is the harder cyclic subset.
DECISION (per the "start 3D soon" steer): BANK this as a scoped, quantified Paper-1 improvement (geometric
opening proposals for the ~44% open misses + cubemap for the 21% near-miss), but do NOT build it now.
PIVOT to the 3D-reconstruction prototype (Paper 2), which is the thesis goal. Come back to the opening
proposer as a bounded (~1 day) headline-lift when convenient.

## Step-0 3D prototype built (Paper 2 kick-off)
`pipelines/gs_room_prototype.py`: GT-posed per-room 3D-Gaussian init from pano+metric-depth
(reuses `sparsepano/gs/gsplat_init.py`), renders the input view (sanity) and a novel view **from
the shared doorway into the neighbour room**, and quantifies disocclusion. Two modes: single-pano
(local smoke test) and ZInD two-room (real Step-0). Deliberately uses GT poses — measures only the
few-view coverage limit, not pose/connectivity.

Design decisions / findings:
- **No gsplat, no GPU for Step-0.** The disocclusion measure is a CPU numpy point-reprojection.
  gsplat/CUDA is deferred to surfel-accurate hole maps / Step-1 optimisation (recipe in
  `docs/Step0_3D_prototype.md`).
- **Metric = point DENSITY, not any-hit coverage.** A point splat has no surface, so far walls leak
  into would-be holes and a naive "any point present?" test saturates at ~1.0 (confirmed: 0.001
  disocclusion regardless of viewpoint). Switched to coarse-cell point counts thresholded at
  `tau_frac × median-cell`; disocclusion = input-well-observed cells that are under-sampled from the
  novel view. Now responds to viewpoint: 0.001 @ 0.3 m step → 0.858 @ 2.5 m in a shallow room.
- **Connected-pair selection.** ZInD annotates a shared door on BOTH room boundaries but the two
  world midpoints differ ~5-10 cm (>the 0.05 m uid tolerance) — so exact-uid matching found ZERO
  cross-room pairs on every floor. Fixed: match by 0.25 m midpoint proximity (validated on 0072/
  0053/0070/0023/0032 metadata).
- **Convention risk flagged.** ZInD `pose_c2w` vs door `endpoints_xy` frame agreement was untestable
  locally (no ZInD panos here). Added a runtime self-check (`az(A->roomB)` vs `az(A->door)`, must
  agree <45°). Must read `convention check: OK` before trusting the cluster number.

### Step-0 first ZInD run (door-vantage, PaGeR depth)
Disocclusion at the doorway vantage (lower bound): 0053=0.056, 0032=0.091, 0023=0.034 (all
convention-OK); 0070=0.104 but flagged **FRAME MISMATCH (154°)** -> not trustworthy. Caveats: the
door is the BEST vantage into room B (underestimates), and the metric is a density proxy, not a
true occlusion hole-map. Next: swept novel-camera along camA->camB (frac 0.25/0.5/0.75) to get the
disocclusion CURVE, and hardened the convention check to use camera-B position + a camA-door-camB
straddle test (centroid-based check gave false alarms on large/L-shaped neighbour rooms). Re-run
the same `scripts/run_gs_prototype.slurm` to get both.

### Step-0 sweep + the real finding (KEY)
Novel-cam swept door→0.25→0.5→0.75 into room B, PaGeR depth. Disocclusion stays LOW and roughly
flat/non-monotonic: 0053 .056/.052/.080/.105 · 0032 .091/.071/.094/.067 · 0023 .034/.021/.033/.023
(0070 was a bad pair — now auto-fixed). **Looking at the renders changes the interpretation:** the
novel views are COHERENT and COMPLETE (e.g. 0053 @0.75 is a clean room with the neighbour visible
back through the door); the dominant artifact is point SPARSITY (dotty), not missing geometry.
=> With one 360° pano per room at GT poses, each room is fully self-observed, so novel views between
adjacent rooms are ~90-98% covered. **Coverage is NOT the Paper-2 bottleneck; the residual a
completion prior would fill is small (grazing/furniture/doorjamb). The sparsity is a RENDERING issue
(real gaussian surfels / gsplat fix it), not a content issue.** This redirects Paper 2 toward the
POSE/alignment quality (which the overlap_probe sparse regime showed is hard: 23.9°/0.271) as the
true determinant of reconstruction quality — dovetails with the connectivity+layout work.

Infra: pair selection now auto-rejects straddle<90 (cameras same side of door) and picks the best
valid door — fixes 0070. Added optional room-B pose-error injection (`--pose_noise_deg/_m`) to run
the pose-sensitivity test (NB: needs a misalignment metric, e.g. masked PSNR novel(GT) vs
novel(noisy) — the density disocclusion won't capture ghosting).

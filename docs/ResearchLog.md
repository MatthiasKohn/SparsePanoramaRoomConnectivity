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

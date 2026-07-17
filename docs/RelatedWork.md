# Related Work

*Positioning notes. Keep entries as: what it does → relevance/difference to us. The 2026 STAR survey
(Pintore et al.) is the primary related-work anchor; adopt its "positioning vs connecting" taxonomy.*

## Foundation-model landscape (2026) — the unifying insight
**All of these need cross-view overlap; none targets our near-zero-overlap, one-pano-per-room regime.**
They differ only in HOW they exploit overlap. Our lane is registration + completion where covisibility
collapses to the doorway.
- **PaGeR** (ETH+Google, 2026; arXiv 2605.26368) — *monocular* panoramic geometry: lifts a perspective
  FM (Depth Anything 3) to 360° via cubemap (6 faces as multi-view), outputs SI+metric depth, normals,
  sky mask in one pass. SOTA single-image geometry. **Not a competitor — adopt as our per-room geometry
  backbone (retire DAP); its normals help door/wall alignment. Does NOT do multi-room registration.**
- **Argus** (Realsee, ECCV'26; 2606.30047) — feed-forward metric multi-view panoramic 3D; needs
  substantial covisibility (covisibility module). Rig-coupled to Realsee capture; did not transfer to
  ZInD in our probe. Overlapping-capture baseline.
- **PanoVGGT** (CVPR'26) — panoramic VGGT; our probe shows it degrades sharply as overlap→0
  (doorway-excepted). The empirical evidence for our motivating claim.
- **IM360** (ICCV'25; 2502.12545) — 360-native SfM: spherical camera model + dense ERP feature matching
  + neural surface + texture. Correspondence-based → needs overlap. Dense-capture competitor, not a
  near-zero-overlap solution.
- **NoPoSplat** (Microsoft, 2410.24207) — pose-free feed-forward 3DGS from sparse *unposed* (overlapping)
  images; canonical-frame anchor, intrinsics-as-token for scale. **Architecture blueprint for our
  Paper-2 completion stage**, to be conditioned on connectivity and pushed to zero overlap.

## Sparse multi-room panorama methods (our direct lineage)
- **[SURVEY] Pintore, Agus, Schneider, Gobbetti — STAR, Eurographics 2026** (DOI 10.1111/cgf.70396;
  repo crs4/panostar): THE survey. Names our subproblem "positioning single rooms" vs "connecting single
  rooms"; Table 9 compares 5 sparse multi-room methods; open challenges (data scarcity, holistic models,
  uncertainty for multiple plausible layouts — the which-side flip is a concrete instance). Cite as
  primary anchor; lift the taxonomy.
- **Extreme SfM** (ICCV'21) — earliest of the line: sparse panos with **no overlap**, single-image layout
  + geometric optimization → coarse floor plan. Origin point of "connectivity from non-overlapping panos."
- **SALVe** (ECCV'22) — W/D/O cues hypothesize pairwise alignments; CNN verifies from BEV renders; GTSAM
  pose graph → 2D floor plan. Our primary 2D baseline. We add: threshold-free connectivity AP as a
  first-class output, the flip-necessity analysis, detector-free-of-layout door embedding.
- **BADGR** (CVPR'25 Highlight; 2503.19340) — diffusion-based bundle adjustment jointly denoising poses +
  1D layouts from wide-baseline sparse panos (as low as 0.6 img/room). **Likely our closest 2025 pose-side
  competitor** (same ZInD lineage). Does not appear to output an explicit connectivity graph — verify by
  full read; that framing may be our difference.
- **Graph-CoVis** (CVPR'23-W) — GNN for co-visible structure + N-view pose on ZInD; pre-diffusion
  alternative to a pose graph. Pose-side baseline.
- **NadirFloorNet** (CVPR'25-W) & **PanoFloor** (ISMAR'25) — reconstruct floor plan / connectivity graph
  but assume **room positions are given as input**. We solve their upstream input (connectivity AND pose
  from pixels alone). Strong positioning line.

## Geometry / GS / completion (Paper-2 relevant)
- **DUSt3R/MASt3R/VGGT family** — feed-forward pose+geometry FMs; documented to fail in low-overlap,
  multi-room, high-symmetry indoor scenes (rooms collapse). Motivating-premise evidence + baselines.
- **Plane-DUSt3R** (ICLR'25) — DUSt3R fine-tuned for multi-view room layout from *unposed perspective*
  views (with overlap). Contrast case for "why not just a 3D FM."
- **Co-VisiON** (ICCV'25-W) — benchmark for "do these sparse views share space"; generalization of our
  connectivity question; cite to justify the problem is recognized.
- **Splatter-360** (CVPR'25) — generalizable 360° GS for **wide-baseline pairs**; still needs shared view.
  Crowded single/paired-room panoramic-GS bucket; doesn't address multi-room stitching.
- **PanoImager** (IROS'26) — SfM-free feed-forward + geometry-conditioned diffusion + depth-guided 3DGS
  for weak-parallax sparse panos (few views of *one* space). Technique family for our completion stage.
- **Pano2Room** (survey Sec 6.1.2) — single pano → mesh → iterative RGB-D inpainting → 3DGS. Single-room;
  its inpaint-and-refuse strategy is relevant to per-room GS disocclusion (exp19 ~29% disoccluded @0.5 m).
- **PanoWorld** (2026) — generative whole-house panorama synthesis from a floorplan (Room-aware Group
  Attention for consistency). Not a competitor (generative, not reconstructive); borrow its cross-room
  consistency mechanism for our walkthrough extension.
- **RoomFormer** — room layout / floorplan generation; source of room-geometry priors.

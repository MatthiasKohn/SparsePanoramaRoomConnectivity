# Semantic Room Connectivity from Sparse 360° Panoramas

Clean restart. Recover room connectivity + relative pose (and later 3D) from one
360° panorama per room, using the geometry visible *through doorways* as the only
cross-room evidence.

## Layout
```
SparsePanoramaRoomConnectivity/
  config.py              # all paths resolved relative to ../data
  src/
    geom.py              # equirect geometry, residual, pose error (validated kernels)
    stanford.py          # Stanford 2D-3D-S loaders + pair / see-through selection
  experiments/
    exp01_pose_ablation.py
  results/exp01_pose_ablation/
  ContextMDs/            # ProjectOverview, ResearchLog, OpenQuestions, PaperNotes
```
Datasets live in `../data` (sibling of this folder): `data/standord2d3d/area_3/...`

## Run
```
python experiments/exp01_pose_ablation.py     # needs: numpy, scipy, opencv, matplotlib
```

## Lightweight viewer
```
python tools/viewer.py --panos path/to/panos
python tools/viewer.py --panos path/to/pano.jpg
python tools/viewer.py --pointcloud results/pointclouds/example.ply
python tools/viewer.py --panos path/to/panos --pointcloud results/pointclouds/example.ply
```

The viewer starts a local browser-based WebGL app, prints a URL, and keeps all
project code untouched. Panoramas are shown as an interactive equirectangular
360 view; point clouds support `.ply`, `.pcd`, `.xyz`, `.txt`, and `.npy` files
with RGB colors when present. Use `--max_points` to downsample large clouds for
interactive debugging.

## Findings so far (see ContextMDs/ResearchLog.md)
- DAP depth is metric and reliable, incl. through doorways (vs Stanford GT). Not the bottleneck.
- Relative rotations between gravity-aligned panos are pure yaw (tilt < ~1°).
- Pose recovery from clean see-through correspondences is well-posed: sub-degree
  rotation, ~3 cm translation. **yaw-only rotation + fixed metric scale** removes
  the divergence tail (0% catastrophic vs ~10% for free Sim(3) at 15° init).
- Next bottleneck: getting clean see-through correspondences WITHOUT ground-truth
  pose (door detection / association) — `exp02`.
```

"""
exp14 — Export colored point clouds (.ply) from panorama depth, for Open3D viewing.

Modes (choose how panos are placed in a common frame):
  (default)   one .ply per pano (camera frame)
  --merge     place all panos by GT pose          (zind / stanford only)
  --estimate  place all panos by ESTIMATED pose   (any dataset; rough, no GT needed)
              -> registers each pano to the anchor (multi-start yaw); shows what the
                 automatic alignment produces. Filter weak ones with --min_inlier.

Pano selection:
  --stems a,b,c   explicit   |   --all   every pano in the dataset (capped by --max)

Examples:
  # full GT-aligned ZInD floor:
  python -m sparsepano.viz.export_pointcloud --dataset zind --all --merge --tint --preview
  # auto-align ALL Immersight panos (no GT) and look at the result:
  python -m sparsepano.viz.export_pointcloud --dataset immersight --all --estimate --tint --preview --max 8
  # one Immersight room:
  python -m sparsepano.viz.export_pointcloud --dataset immersight --stems 1273546

  # view:  o3d.io.read_point_cloud("results/pointclouds/<name>.ply")
"""
import sys, os, argparse
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.geometry import geom

OUT = config.RESULTS_ROOT / "pointclouds"; OUT.mkdir(parents=True, exist_ok=True)
IMM = config.DATA_ROOT / "Download_immersight_ 2026-06-23_10-23-49"
TINTS = np.array([[230, 75, 75], [75, 150, 230], [80, 200, 120], [235, 195, 70],
                  [180, 110, 235], [90, 200, 210], [240, 140, 70], [150, 150, 150]])


def write_ply(path, xyz, rgb):
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    data = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                              ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['r'], data['g'], data['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode()); f.write(data.tobytes())


def cloud_from(depth, rgb, stride=3, max_depth=12.0, max_points=200000, rng=None):
    H, W = depth.shape
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    pts, us, vs = geom.backproject(depth, stride=stride)
    keep = np.linalg.norm(pts, axis=1) < max_depth
    pts, us, vs = pts[keep], us[keep].astype(int), vs[keep].astype(int)
    cols = rgb[vs, us]
    if len(pts) > max_points:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols


# ---- dataset loaders: stem -> (depth, rgb) ; provider for GT poses ----
def immersight_load(stem):
    d = np.load(IMM / "dap_depth" / "depth_metric" / f"panorama_{stem}.npy").astype(np.float32)
    for ext in (".png", ".jpeg", ".jpg"):
        p = IMM / f"panorama_{stem}{ext}"
        if p.exists():
            return d, cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    return d, None


def get_provider(dataset):
    from sparsepano.geometry import providers
    if dataset == "stanford":
        return providers.StanfordProvider()
    if dataset == "zind":
        return providers.default_zind()
    return None


def load(dataset, stem, prov):
    if dataset == "immersight":
        return immersight_load(stem)
    if dataset == "stanford":
        from sparsepano.datasets import stanford as st
        rgb = cv2.cvtColor(cv2.imread(str(prov.P["rgb"] / f"{stem}_rgb.png")), cv2.COLOR_BGR2RGB)
        return st.load_dap_depth(stem, prov.P, rgb.shape[:2]), rgb
    if dataset == "zind":
        rgb = cv2.cvtColor(cv2.imread(str(config.zind_paths()["panos"] / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        return prov.depth(stem), rgb


def discover(dataset, prov):
    if dataset == "immersight":
        return sorted({p.stem.replace("panorama_", "") for p in IMM.glob("panorama_*")
                       if p.suffix.lower() in (".png", ".jpeg", ".jpg")})
    if dataset == "zind":
        return sorted(s.stem for s in prov.depth_dir.glob("*.npy") if s.stem in prov.fl.panos)
    if dataset == "stanford":
        return [n for n in prov.names if (prov.P["dap_depth"] / f"{n}_rgb.npy").exists()]


def estimate_pose(target_depth, anchor_depth, rng):
    """Register target -> anchor (multi-start yaw). Returns (R,t,s,inlier)."""
    from sparsepano.pose import register
    da = cv2.resize(anchor_depth, (2048, 1024), interpolation=cv2.INTER_NEAREST)
    dt = cv2.resize(target_depth, (2048, 1024), interpolation=cv2.INTER_NEAREST)
    pts, _, _ = geom.backproject(dt, stride=3)
    if len(pts) > 6000:
        pts = pts[rng.choice(len(pts), 6000, replace=False)]
    best = register.register(pts, da, 2048, 1024, yaw_steps=18)
    return best["R"], best["t"], best["s"], best["inlier"]


def main(a):
    prov = get_provider(a.dataset)
    stems = a.stems.split(",") if a.stems else discover(a.dataset, prov)
    stems = stems[:a.max]
    mode = "estimate" if a.estimate else ("merge" if a.merge else "single")
    if mode == "merge" and prov is None:
        print("--merge needs GT poses (zind/stanford); use --estimate for immersight"); return
    print(f"{a.dataset}: {len(stems)} panos, mode={mode}")
    rng = np.random.default_rng(0)
    anchor = stems[0]
    anchor_depth = load(a.dataset, anchor, prov)[0] if mode == "estimate" else None

    clouds = []
    for k, s in enumerate(stems):
        depth, rgb = load(a.dataset, s, prov)
        if depth is None or rgb is None:
            print(f"  [skip] {s}"); continue
        xyz, col = cloud_from(depth, rgb, stride=a.stride, max_points=a.max_points, rng=rng)
        tag = s[-14:]
        if mode == "merge":
            T = prov.rel_pose(anchor, s); R, t = T[:3, :3], T[:3, 3]
            xyz = (xyz - t) @ R
        elif mode == "estimate":
            if s != anchor:
                R, t, sc, inl = estimate_pose(depth, anchor_depth, rng)
                if inl < a.min_inlier:
                    print(f"  [drop] {tag} inlier {inl:.2f} < {a.min_inlier}"); continue
                xyz = sc * (xyz @ R.T) + t
                print(f"  {tag}: inlier {inl:.2f}  |t| {np.linalg.norm(t):.2f}m")
            else:
                print(f"  {tag}: anchor")
        if (a.tint and mode != "single"):
            col = (0.45 * col + 0.55 * TINTS[k % len(TINTS)]).astype(np.uint8)
        if mode == "single":
            p = OUT / f"{a.dataset}_{s}.ply"; write_ply(p, xyz, col)
            print(f"  wrote {p}  ({len(xyz):,} pts)")
        clouds.append((xyz, col))

    if mode != "single" and clouds:
        XYZ = np.concatenate([c[0] for c in clouds]); COL = np.concatenate([c[1] for c in clouds])
        # pose-source tag so the viewer header is unambiguous: GT poses vs pipeline estimate
        POSE_TAG = {"merge": "GTpose", "estimate": "PIPELINEpose", "single": "single"}
        name = f"{a.dataset}_{POSE_TAG.get(mode, mode)}_{len(clouds)}panos.ply"
        write_ply(OUT / name, XYZ, COL); print(f"\n  wrote {OUT/name}  ({len(XYZ):,} pts)")
        if a.preview:
            _preview(clouds, OUT / name.replace(".ply", "_topdown.png"))
    elif a.preview and clouds:
        _preview(clouds, OUT / f"{a.dataset}_{stems[0]}_topdown.png")


def _preview(clouds, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    for xyz, col in clouds:
        m = (xyz[:, 1] > -1.0) & (xyz[:, 1] < 0.8)
        ax.scatter(xyz[m, 0], xyz[m, 2], s=0.4, c=col[m] / 255.0)
    ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title("top-down preview"); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig); print(f"  preview {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["immersight", "zind", "stanford"], default="immersight")
    ap.add_argument("--stems", help="comma-separated stems (omit to use --all)")
    ap.add_argument("--all", action="store_true", help="use every pano in the dataset")
    ap.add_argument("--max", type=int, default=12, help="cap number of panos")
    ap.add_argument("--merge", action="store_true", help="place all by GT pose (zind/stanford)")
    ap.add_argument("--estimate", action="store_true", help="place all by ESTIMATED pose (any dataset)")
    ap.add_argument("--min_inlier", type=float, default=0.0, help="drop estimated panos below this overlap")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_points", type=int, default=200000)
    ap.add_argument("--tint", action="store_true")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    if not a.stems and not getattr(a, "all"):
        ap.error("give --stems or --all")
    main(a)

"""
Per-room 3D Gaussian initialization from ONE panorama + its metric depth (E0).

A single 360 pano + metric monocular depth gives a colored 3D point cloud; we turn
each point into a 3D Gaussian (position, colour, opacity, scale, identity rotation).
Scale is initialized to the world-space footprint of one pano pixel at that depth
(depth * angular-pixel-size) -- a physically meaningful, library-agnostic init.

CPU here (numpy). Two outputs:
  - a 3DGS-format .ply (f_dc / opacity / scale / rotation) to seed a GS optimizer/viewer;
  - render_equirect(): reproject the Gaussian centers back to the pano for a SANITY check
    (same view should reproduce the input; a small-baseline view shows disocclusion holes
    -- the single-view limitation that GS optimization / priors must fill).

The actual differentiable GS optimization (CUDA rasterizer) is isolated in exp19's
`optimize_room_gs` hook -- run on a GPU machine.
"""
import numpy as np
import cv2
from sparsepano.geometry import geom

SH_C0 = 0.28209479177387814          # SH degree-0 basis (3DGS colour convention)


def gaussian_init_from_pano(depth, rgb, stride=2, max_depth=12.0, scale_mult=1.0):
    H, W = depth.shape
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    pts, us, vs = geom.backproject(depth, stride=stride)
    r = np.linalg.norm(pts, axis=1)
    keep = (r > 0.1) & (r < max_depth)
    pts, us, vs, r = pts[keep], us[keep].astype(int), vs[keep].astype(int), r[keep]
    col = rgb[vs, us].astype(np.float32) / 255.0
    px = 2 * np.pi / W                                   # angular size of one column (rad)
    scale = (scale_mult * np.clip(r * px, 1e-3, None))[:, None] * np.ones((1, 3), np.float32)
    return dict(xyz=pts.astype(np.float32), rgb=col.astype(np.float32),
                opacity=np.full(len(pts), 0.99, np.float32),
                scale=scale.astype(np.float32),
                rot=np.tile([1.0, 0, 0, 0], (len(pts), 1)).astype(np.float32))


def _logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def write_gs_ply(path, g):
    """3DGS-format binary .ply (loadable by gaussian-splatting / SuperSplat viewers)."""
    n = len(g["xyz"])
    fdc = (g["rgb"] - 0.5) / SH_C0
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros((n, len(props)), np.float32)
    arr[:, 0:3] = g["xyz"]
    arr[:, 6:9] = fdc
    arr[:, 9] = _logit(g["opacity"])
    arr[:, 10:13] = np.log(np.clip(g["scale"], 1e-8, None))
    arr[:, 13:17] = g["rot"]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode()); f.write(arr.astype("<f4").tobytes())


def write_point_ply(path, g):
    """Plain colored point cloud (xyz + rgb), for quick Open3D viewing."""
    n = len(g["xyz"]); rgb = (np.clip(g["rgb"], 0, 1) * 255).astype(np.uint8)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    data = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                              ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    data['x'], data['y'], data['z'] = g["xyz"].T
    data['r'], data['g'], data['b'] = rgb.T
    with open(path, "wb") as f:
        f.write(header.encode()); f.write(data.tobytes())


def render_equirect(g, H, W, R=None, t=None):
    """Reproject Gaussian centers to an equirect view (z-buffered points). Sanity only."""
    xyz = g["xyz"]
    if R is not None or t is not None:
        R = np.eye(3) if R is None else R
        t = np.zeros(3) if t is None else np.asarray(t, float)
        xyz = xyz @ R.T + t
    u, v, r = geom.project_to_pano(xyz, W, H)
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    img = np.zeros((H, W, 3), np.float32); mask = np.zeros((H, W), bool)
    o = np.argsort(-r)                                   # far first; near overwrites
    img[vi[o], ui[o]] = g["rgb"][o]; mask[vi[o], ui[o]] = True
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), mask


def psnr(a, b, mask=None):
    a = a.astype(np.float32); b = b.astype(np.float32)
    if mask is not None:
        a, b = a[mask], b[mask]
    mse = np.mean((a - b) ** 2)
    return float(99.0 if mse < 1e-6 else 10 * np.log10(255.0 ** 2 / mse))

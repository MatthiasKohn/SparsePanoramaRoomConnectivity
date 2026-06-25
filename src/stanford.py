"""Stanford 2D-3D-S loaders and connected-pair selection."""
import json
import numpy as np
import cv2
from itertools import combinations

import config
from src import geom


def list_panos(area="area_3"):
    P = config.stanford_area(area)
    names = [p.name.replace("_pose.json", "") for p in sorted(P["pose"].glob("*_pose.json"))]
    return names, P


def load_pose(name, P):
    with open(P["pose"] / f"{name}_pose.json") as f:
        d = json.load(f)
    rt = np.array(d["camera_rt_matrix"])
    return {"R": rt[:, :3], "t": rt[:, 3],
            "loc": np.array(d["camera_location"]),
            "room": d.get("room", ""), "name": name}


def get_hw(name, P):
    img = cv2.imread(str(P["rgb"] / f"{name}_rgb.png"))
    return None if img is None else img.shape[:2]


def load_dap_depth(name, P, hw=None):
    d = np.load(P["dap_depth"] / f"{name}_rgb.npy").astype(np.float32)
    if hw and d.shape != tuple(hw):
        d = cv2.resize(d, (hw[1], hw[0]))
    return d


def load_gt_depth(name, P, hw=None):
    g = cv2.imread(str(P["gt_depth"] / f"{name}_rgb_depth.png"), cv2.IMREAD_UNCHANGED)
    if g is None:  # some releases name it *_depth.png without _rgb
        g = cv2.imread(str(P["gt_depth"] / f"{name}_depth.png"), cv2.IMREAD_UNCHANGED)
    g = g.astype(np.float32)
    invalid = g >= config.STANFORD_INVALID
    g = g * config.STANFORD_GT_DEPTH_UNIT
    g[invalid] = np.nan
    if hw and g.shape != tuple(hw):
        g = cv2.resize(g, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return g


def connected_pairs(poses, max_dist=2.5):
    """Proxy connectivity: different room AND camera centres within max_dist.
    Not true GT connectivity, but a clean set of clearly-adjacent pairs for a
    pose-conditioning study. Returns list of (name_a, name_b, dist)."""
    out = []
    for na, nb in combinations(poses.keys(), 2):
        pa, pb = poses[na], poses[nb]
        if pa["room"] == pb["room"]:
            continue
        dist = float(np.linalg.norm(pa["loc"] - pb["loc"]))
        if dist < max_dist:
            out.append((na, nb, dist))
    out.sort(key=lambda x: x[2])
    return out


def select_shared_points(depth_a, depth_b, T_gt, W, H, stride=6, tau=0.25,
                         max_pts=1500, min_depth=0.3, rng=None):
    """Genuine co-visible ('see-through') A-points: backproject A, map to B at
    GT pose, keep those whose predicted range matches B's measured depth within
    tau. Selection uses GT pose; the optimiser later must RECOVER pose from a
    perturbed start, so this is not circular w.r.t. the convergence test."""
    pts_a, us, vs = geom.backproject(depth_a, stride=stride)
    if len(pts_a) == 0:
        return None
    R, t = T_gt[:3, :3], T_gt[:3, 3]
    pb = pts_a @ R.T + t
    u, v, r = geom.project_to_pano(pb, W, H)
    meas = geom.sample_bilinear(depth_b, u, v)
    a_range = np.linalg.norm(pts_a, axis=1)
    ok = (np.isfinite(meas) & (meas > 0.1) & (a_range > min_depth)
          & (np.abs(r - meas) < tau))
    pts = pts_a[ok]
    if len(pts) < 50:
        return None
    if len(pts) > max_pts:
        rng = rng or np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    return pts

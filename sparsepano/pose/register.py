"""
Unknown-pose pairwise registration of two panoramas (no GT init).

Gravity-aligned assumption => relative rotation is a single yaw. We multi-start
over yaw (translation 0), refine each start with the V2 depth-consistency
optimiser, and keep the start with the highest inlier fraction. With substantial
overlap the correct yaw wins cleanly; the inlier fraction doubles as an overlap /
confidence score (no GT needed).
"""
import numpy as np
import cv2

from sparsepano import config
from sparsepano.geometry import geom
from sparsepano.pose import pose as posemod


def load_depth(stem, depth_dir, target_h=1024):
    d = np.load(depth_dir / f"{stem}.npy").astype(np.float32)
    if d.shape[0] != target_h:
        d = cv2.resize(d, (target_h * 2, target_h), interpolation=cv2.INTER_NEAREST)
    return d


def inlier_fraction(pts, depth_b, R, t, s, W, H, tau=0.15):
    res, valid = geom.residual(pts, depth_b, R, t, s, W, H)
    if res.size == 0:
        return 0.0, 0.0
    inl = np.abs(res) < tau
    # fraction over points that actually fell inside B's valid depth
    return float(inl.mean()), float(valid.mean())


def register(pts_a, depth_b, W, H, yaw_steps=24, tau=0.15, max_nfev=120):
    best = None
    for yaw0 in np.linspace(0, 2 * np.pi, yaw_steps, endpoint=False):
        Ti = np.eye(4)
        Ti[:3, :3] = geom.Ry(yaw0)
        out = posemod.recover(pts_a, depth_b, Ti, W, H, variant="V2", max_nfev=max_nfev)
        frac, cover = inlier_fraction(pts_a, depth_b, out["R"], out["t"], out["s"], W, H, tau)
        score = frac * cover                       # reward consistent AND covered
        if best is None or score > best["score"]:
            best = dict(R=out["R"], t=out["t"], s=out["s"], yaw0=yaw0,
                        inlier=frac, cover=cover, score=score, resid=out["resid"])
    return best

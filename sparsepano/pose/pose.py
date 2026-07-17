"""Pose recovery from see-through correspondences.

Adopted model (from exp01): V2 = yaw-only rotation + fixed metric scale=1.
Also exposes V0/V1 for ablation. Returns the converged robust residual so a
caller can use it to VERIFY / disambiguate aperture candidates.
"""
import numpy as np
from scipy.optimize import least_squares

from sparsepano.geometry import geom

HUBER = 0.3


def _model(variant, T_init):
    R0, t0 = T_init[:3, :3].copy(), T_init[:3, 3].copy()
    if variant == "V0":
        x0 = np.zeros(7)
        lo = np.array([-np.inf]*6 + [np.log(0.5)]); hi = np.array([np.inf]*6 + [np.log(2.0)])
        unpack = lambda x: (R0 @ geom.so3_exp(x[:3]), t0 + R0 @ x[3:6], np.exp(x[6]))
    elif variant == "V1":
        x0 = np.zeros(5)
        lo = np.array([-np.inf]*4 + [np.log(0.5)]); hi = np.array([np.inf]*4 + [np.log(2.0)])
        unpack = lambda x: (R0 @ geom.Ry(x[0]), t0 + R0 @ x[1:4], np.exp(x[4]))
    else:  # V2
        x0 = np.zeros(4)
        lo = np.full(4, -np.inf); hi = np.full(4, np.inf)
        unpack = lambda x: (R0 @ geom.Ry(x[0]), t0 + R0 @ x[1:4], 1.0)
    return x0, lo, hi, unpack


def robust_mean(res):
    a = np.abs(res); d = HUBER
    return float(np.mean(np.where(a < d, 0.5 * res**2, d * (a - 0.5 * d)))) if res.size else 1e3


def recover(pts, depth_b, T_init, W, H, variant="V2", max_nfev=200):
    """Returns dict with R,t,s and converged robust residual (lower = better fit)."""
    x0, lo, hi, unpack = _model(variant, T_init)
    n = pts.shape[0]

    def resfun(x):
        R, t, s = unpack(x)
        res, _ = geom.residual(pts, depth_b, R, t, s, W, H)
        out = np.zeros(n)
        if res.size >= 10:
            out[:res.size] = res
        else:
            out[:] = 10.0
        return out

    sol = least_squares(resfun, x0, bounds=(lo, hi), loss="huber",
                        f_scale=HUBER, max_nfev=max_nfev, method="trf")
    R, t, s = unpack(sol.x)
    final_res, _ = geom.residual(pts, depth_b, R, t, s, W, H)
    return {"R": R, "t": t, "s": s, "resid": robust_mean(final_res), "n": n}

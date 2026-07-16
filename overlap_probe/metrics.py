"""
overlap_probe.metrics — convention-robust pose evaluation.

Feed-forward models return camera poses in their OWN world frame, at their own scale (unless
metric), and with an arbitrary global rotation. We therefore evaluate with quantities that are
invariant to that gauge freedom:

  * ATE (Sim3-aligned): Umeyama-align predicted camera CENTRES to GT (rotation+translation
    +scale), then RMSE of the residual, in meters. Also reported normalized by scene diameter
    so scenes of different size are comparable. The fitted scale s is reported too: for a
    truly METRIC model (Argus) s should be ~1; |log s| is its metric-scale error.

  * Relative-rotation error: for every pano PAIR, geodesic angle between the predicted relative
    rotation R_i^T R_j and the GT one. Relative rotations are invariant to the global frame, so
    this needs NO alignment and NO scale — the cleanest cross-model rotation metric.

All angles in degrees, distances in meters.
"""
import numpy as np
from itertools import combinations


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity mapping src->dst. src,dst: (N,3). Returns (s,R,t)."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    Sigma = (Dc.T @ Sc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (Sc ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / (var_s + 1e-12))
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def geodesic_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def eval_poses(pred_c2w, gt_c2w):
    """pred_c2w, gt_c2w: (N,4,4). Returns a metrics dict (NaN-safe)."""
    N = len(gt_c2w)
    Cp = pred_c2w[:, :3, 3]; Cg = gt_c2w[:, :3, 3]
    Rp = pred_c2w[:, :3, :3]; Rg = gt_c2w[:, :3, :3]

    diam = 0.0
    for i, j in combinations(range(N), 2):
        diam = max(diam, float(np.linalg.norm(Cg[i] - Cg[j])))
    diam = max(diam, 1e-6)

    out = {"n": N, "diam_m": diam}

    if N >= 3:
        s, R, t = umeyama(Cp, Cg, with_scale=True)
        aligned = (s * (R @ Cp.T).T) + t
        ate = float(np.sqrt(((aligned - Cg) ** 2).sum(1).mean()))
        out["ate_m"] = ate
        out["ate_norm"] = ate / diam
        out["sim3_scale"] = s
        out["logscale_abs"] = float(abs(np.log(s))) if s > 0 else float("nan")
    else:
        out.update(ate_m=float("nan"), ate_norm=float("nan"),
                   sim3_scale=float("nan"), logscale_abs=float("nan"))

    # relative rotation errors, per pair (alignment-free); also keep per-pair for stratification
    per_pair = {}
    errs = []
    for i, j in combinations(range(N), 2):
        e = geodesic_deg(Rg[i].T @ Rg[j], Rp[i].T @ Rp[j])
        per_pair[(i, j)] = e
        errs.append(e)
    out["relrot_med_deg"] = float(np.median(errs)) if errs else float("nan")
    out["relrot_mean_deg"] = float(np.mean(errs)) if errs else float("nan")
    out["_per_pair_relrot"] = per_pair
    return out


def stratify_relrot(metrics, scene):
    """Median relative-rotation error split by overlap category."""
    buckets = {"same": [], "adjacent": [], "far": []}
    for pair, e in metrics["_per_pair_relrot"].items():
        cat = scene.overlap.get(pair, ("far", 0.0))[0]
        buckets[cat].append(e)
    return {k: (float(np.median(v)) if v else float("nan"), len(v))
            for k, v in buckets.items()}

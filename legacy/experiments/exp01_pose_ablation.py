"""
exp01 — Pose-recovery ablation: is Stage-2's failure a parameterization bug?

Hypothesis (from diagnosing the old Stage-2 code):
  Rotation error spread 5-95 deg even from a 5 deg start because the Sim(3)
  optimiser has (a) a free metric scale (verified ~1.0) and (b) full SO(3)
  rotation, while gravity-aligned panoramas only differ by a yaw. These extra
  DOF open a near-flat valley in a thin radial see-through residual.

Test: same see-through points, same yaw+translation perturbations, three
optimisers of increasing constraint:
  V0  full Sim(3)            : SO(3) rot + 3D trans + free scale   (old method)
  V1  yaw + free scale       : isolate effect of rotation parameterization
  V2  yaw + fixed scale=1    : add the metric-scale fix

Prediction: V0 -> V1 collapses rotation error; V1 -> V2 tightens translation.

Run:  python legacy/experiments/exp01_pose_ablation.py
"""
import sys, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from sparsepano import config
from sparsepano.geometry import geom
from sparsepano.datasets import stanford

AREA = "area_3"
MAX_DIST = 2.5
MAX_PAIRS = 10
PERTURBS = [(5.0, 0.3), (15.0, 0.5)]   # (yaw deg, translation m)
N_TRIALS = 2
HUBER = 0.3
SEED = 0
OUT = config.RESULTS_ROOT / "exp01_pose_ablation"
OUT.mkdir(parents=True, exist_ok=True)


def perturb_yaw(T_gt, yaw_deg, trans_m, rng):
    a = np.deg2rad(yaw_deg) * rng.choice([-1, 1])
    dR = geom.Ry(a)
    dt = rng.normal(size=3); dt = dt / np.linalg.norm(dt) * trans_m
    T = T_gt.copy()
    T[:3, :3] = T_gt[:3, :3] @ dR
    T[:3, 3] = T_gt[:3, 3] + dt
    return T


def make_model(variant, T_init):
    R0, t0 = T_init[:3, :3].copy(), T_init[:3, 3].copy()
    if variant == "V0":          # full Sim(3)
        x0 = np.zeros(7)
        lo = np.array([-np.inf]*6 + [np.log(0.5)])
        hi = np.array([ np.inf]*6 + [np.log(2.0)])
        def unpack(x):
            return R0 @ geom.so3_exp(x[:3]), t0 + R0 @ x[3:6], np.exp(x[6])
    elif variant == "V1":        # yaw + free scale
        x0 = np.zeros(5)
        lo = np.array([-np.inf]*4 + [np.log(0.5)])
        hi = np.array([ np.inf]*4 + [np.log(2.0)])
        def unpack(x):
            return R0 @ geom.Ry(x[0]), t0 + R0 @ x[1:4], np.exp(x[4])
    else:                        # V2 yaw + fixed scale
        x0 = np.zeros(4)
        lo = np.full(4, -np.inf); hi = np.full(4, np.inf)
        def unpack(x):
            return R0 @ geom.Ry(x[0]), t0 + R0 @ x[1:4], 1.0
    return x0, lo, hi, unpack


def fit(variant, pts, depth_b, T_init, T_gt, W, H):
    x0, lo, hi, unpack = make_model(variant, T_init)

    def resfun(x):
        R, t, s = unpack(x)
        res, valid = geom.residual(pts, depth_b, R, t, s, W, H)
        if res.size < 10:
            return np.full(pts.shape[0], 10.0)
        # pad/truncate to fixed length so least_squares sees constant size
        out = np.full(pts.shape[0], 0.0)
        out[:res.size] = res
        return out

    sol = least_squares(resfun, x0, bounds=(lo, hi), loss="huber",
                        f_scale=HUBER, max_nfev=200, method="trf")
    R, t, s = unpack(sol.x)
    rot, trans = geom.pose_error(R, t, T_gt)
    return rot, trans, s


def robust_mean(res):
    a = np.abs(res); d = HUBER
    return np.mean(np.where(a < d, 0.5 * res**2, d * (a - 0.5 * d)))


def cost_landscape(pts, depth_b, T_gt, W, H):
    """Residual vs single-DOF perturbation around GT: yaw, off-yaw(X), scale."""
    R0, t0 = T_gt[:3, :3], T_gt[:3, 3]
    angs = np.linspace(-30, 30, 61)
    yaw_c, offyaw_c = [], []
    for a in np.deg2rad(angs):
        R = R0 @ geom.Ry(a)
        res, _ = geom.residual(pts, depth_b, R, t0, 1.0, W, H)
        yaw_c.append(robust_mean(res))
        Rx = R0 @ geom.so3_exp(np.array([a, 0, 0]))
        res2, _ = geom.residual(pts, depth_b, Rx, t0, 1.0, W, H)
        offyaw_c.append(robust_mean(res2))
    scales = np.linspace(0.5, 1.6, 56)
    sc_c = []
    for s in scales:
        res, _ = geom.residual(pts, depth_b, R0, t0, s, W, H)
        sc_c.append(robust_mean(res))
    return angs, np.array(yaw_c), np.array(offyaw_c), scales, np.array(sc_c)


def main():
    rng = np.random.default_rng(SEED)
    names, P = stanford.list_panos(AREA)
    poses = {n: stanford.load_pose(n, P) for n in names}
    pairs = stanford.connected_pairs(poses, MAX_DIST)[:MAX_PAIRS]
    print(f"Area {AREA}: {len(poses)} panos, {len(pairs)} connected-proxy pairs (<{MAX_DIST} m)")

    # --- premise check: are relative rotations actually yaw (gravity aligned)? ---
    tilts = []
    for na, nb, _ in stanford.connected_pairs(poses, MAX_DIST):
        T = geom.rel_pose(poses[na], poses[nb])
        _, tilt = geom.decompose_gravity(T[:3, :3])
        tilts.append(tilt)
    tilts = np.array(tilts)
    print(f"[premise] vertical-axis tilt of relative rotation: "
          f"median={np.median(tilts):.2f} deg, 95%={np.percentile(tilts,95):.2f} deg, "
          f"max={tilts.max():.2f} deg  (small => yaw-only justified)")

    rows = []
    H = W = None
    landscape_pair = None; landscape_npts = -1
    for na, nb, dist in pairs:
        hw = stanford.get_hw(na, P)
        if hw is None:
            continue
        H, W = hw
        da = stanford.load_dap_depth(na, P, hw)
        db = stanford.load_dap_depth(nb, P, hw)
        T_gt = geom.rel_pose(poses[na], poses[nb])
        pts = stanford.select_shared_points(da, db, T_gt, W, H, rng=rng)
        if pts is None:
            print(f"  [skip] {poses[na]['room']}<->{poses[nb]['room']}: too few shared pts")
            continue
        if len(pts) > landscape_npts:
            landscape_npts = len(pts)
            landscape_pair = (pts.copy(), db, T_gt)
        for (yaw, tm) in PERTURBS:
            for _ in range(N_TRIALS):
                T_init = perturb_yaw(T_gt, yaw, tm, rng)
                for V in ("V0", "V1", "V2"):
                    rot, trans, s = fit(V, pts, db, T_init, T_gt, W, H)
                    rows.append(dict(pair=f"{poses[na]['room']}|{poses[nb]['room']}",
                                     dist=dist, npts=len(pts), yaw=yaw, tm=tm,
                                     variant=V, rot_err=rot, trans_err=trans, scale=s))
        print(f"  {poses[na]['room']:>14}<->{poses[nb]['room']:<14} d={dist:.2f} "
              f"npts={len(pts)}")

    import csv
    with open(OUT / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader()
        wr.writerows(rows)

    # --- summary table ---
    print("\n" + "=" * 64)
    print(f"{'variant':8} {'rot med':>8} {'rot<5d%':>8} {'trans med':>10} {'trans<.3%':>10}")
    for V in ("V0", "V1", "V2"):
        r = np.array([x["rot_err"] for x in rows if x["variant"] == V])
        t = np.array([x["trans_err"] for x in rows if x["variant"] == V])
        print(f"{V:8} {np.median(r):8.2f} {100*np.mean(r<5):8.0f} "
              f"{np.median(t):10.2f} {100*np.mean(t<0.3):10.0f}")
    print("=" * 64)

    # --- figure 1: error distributions ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, key in enumerate(["rot_err", "trans_err"]):
        data = [[x[key] for x in rows if x["variant"] == V] for V in ("V0", "V1", "V2")]
        ax[i].boxplot(data, tick_labels=["V0 full-Sim3", "V1 yaw+freeS", "V2 yaw+fixS"],
                      showfliers=True)
        ax[i].set_title("Rotation error (deg)" if i == 0 else "Translation error (m)")
        ax[i].grid(alpha=0.3)
    fig.suptitle("exp01: pose-recovery ablation (Stanford area_3, DAP depth)")
    fig.tight_layout(); fig.savefig(OUT / "error_distributions.png", dpi=130)

    # --- figure 2: cost landscape ---
    if landscape_pair is not None:
        pts, db, T_gt = landscape_pair
        angs, yaw_c, offyaw_c, scales, sc_c = cost_landscape(pts, db, T_gt, W, H)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot(angs, yaw_c, label="yaw (about up)")
        ax[0].plot(angs, offyaw_c, label="off-yaw (about X)", ls="--")
        ax[0].axvline(0, color="k", lw=0.7); ax[0].set_xlabel("rotation offset (deg)")
        ax[0].set_ylabel("robust mean residual"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[0].set_title("Rotation conditioning around GT")
        ax[1].plot(scales, sc_c); ax[1].axvline(1.0, color="k", lw=0.7)
        ax[1].set_xlabel("scale"); ax[1].set_ylabel("robust mean residual")
        ax[1].set_title("Scale conditioning around GT"); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "cost_landscape.png", dpi=130)

    print(f"\nWrote: {OUT}/results.csv, error_distributions.png, cost_landscape.png")


if __name__ == "__main__":
    main()

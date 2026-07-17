"""
exp21 (E2) — Two-room fusion with the ESTIMATED door-anchored pose + the flip choice.

E1 used the GT pose. Here we use the door-anchored MEASURED pose (sparsepano/pose/door_pose) and show
the 3D consequence of the which-side flip:
  - CORRECT side  -> rooms sit adjacent across the doorway (coherent join, like GT);
  - WRONG (flip)  -> room B is reflected onto the wrong side and COLLIDES with room A.
The embedding flip-prior (exp18, 5/6) is what selects the correct side; this experiment
shows why that choice is decisive for the 3D reconstruction.

Metric: inter-room interpenetration = fraction of B's floor points that fall INSIDE room A
(should be low for the correct side, high for the flip).

  python legacy/experiments/exp21_gs_pair_estimated.py --a <stemA> --b <stemB>
"""
import sys, os, argparse
from itertools import combinations
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.geometry import providers
from sparsepano.geometry import geom
from sparsepano.pose import door_pose
from sparsepano.gs import gsplat_init as gsi


def load_pano(prov, stem, hw):
    im = cv2.imread(str(prov.pano_dir / f"{stem}.jpg"))
    return cv2.cvtColor(cv2.resize(im, (hw[1], hw[0])), cv2.COLOR_BGR2RGB)


def transform(g, R, t, s=1.0):
    out = dict(g); out["xyz"] = (1.0 / s) * (g["xyz"] @ R) + t
    return out


def flip_about(g, door):
    out = dict(g); xyz = g["xyz"].copy()
    xyz[:, 0] = 2 * door[0] - xyz[:, 0]; xyz[:, 2] = 2 * door[2] - xyz[:, 2]
    out["xyz"] = xyz; return out


def wall_profile_pts(xyz, nbins=72):
    r = np.linalg.norm(xyz[:, [0, 2]], axis=1)
    rr = np.linalg.norm(xyz, axis=1) + 1e-9
    el = np.degrees(np.arcsin(np.clip(xyz[:, 1] / rr, -1, 1)))
    m = np.abs(el) < 25
    az = np.arctan2(xyz[m, 0], xyz[m, 2]); r = r[m]
    b = ((az + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    prof = np.full(nbins, np.nan)
    for k in range(nbins):
        s = r[b == k]
        if len(s) > 3:
            prof[k] = np.percentile(s, 60)
    return prof


def interpenetration(profA, xyzB, nbins=72):
    r = np.linalg.norm(xyzB[:, [0, 2]], axis=1)
    rr = np.linalg.norm(xyzB, axis=1) + 1e-9
    el = np.degrees(np.arcsin(np.clip(xyzB[:, 1] / rr, -1, 1)))
    m = np.abs(el) < 25
    az = np.arctan2(xyzB[m, 0], xyzB[m, 2]); r = r[m]
    b = ((az + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    wall = profA[b]; ok = np.isfinite(wall)
    return float((r[ok] < 0.85 * wall[ok]).mean()) if ok.sum() else float("nan")


def auto_pair(prov):
    names = [n for n in prov.fl.panos if (prov.depth_dir / f"{n}.npy").exists()]
    for a, b in combinations(names, 2):
        if prov.fl.panos[a]["room"] != prov.fl.panos[b]["room"] \
           and prov.fl.shared_door(a, b) is not None \
           and prov.shared_door_bearing(a, b) is not None:
            return a, b
    return names[0], names[1]


def main(a):
    prov = providers.default_zind()
    H, W = a.res, a.res * 2
    sa, sb = (a.a, a.b) if a.a and a.b else auto_pair(prov)
    print(f"pair: {sa}  <->  {sb}")
    dA = cv2.resize(prov.depth(sa), (W, H), interpolation=cv2.INTER_NEAREST)
    dB = cv2.resize(prov.depth(sb), (W, H), interpolation=cv2.INTER_NEAREST)
    gA = gsi.gaussian_init_from_pano(dA, load_pano(prov, sa, (H, W)), stride=a.stride)
    gB = gsi.gaussian_init_from_pano(dB, load_pano(prov, sb, (H, W)), stride=a.stride)

    # measured door-anchored pose (A->B): p_B = s R p_A + t
    az_a = prov.shared_door_bearing(sa, sb); az_b = prov.shared_door_bearing(sb, sa)
    o = door_pose.recover(dA, dB, np.degrees(az_a), np.degrees(az_b), W, H)
    R, t, s = o["R"], o["t"], o["s"]
    cam = -(1.0 / s) * (R.T @ t)                                   # B camera in A frame
    wall = door_pose._sector_wall(dA, np.degrees(az_a), W, H)[1]
    door = np.array([wall * np.sin(az_a), 0.0, wall * np.cos(az_a)])   # door point in A (3D)
    print(f"  door-anchored inlier {o['inlier']:.2f}  |cam_B| {np.linalg.norm(cam):.2f} m")

    gB_meas = transform(gB, R, cam, s)            # estimated correct-side placement
    gB_flip = flip_about(gB_meas, door)           # which-side flip (pi about door vertical)
    Tba = prov.rel_pose(sb, sa); gB_gt = transform(gB, Tba[:3, :3].T, Tba[:3, 3], 1.0) \
        if False else {**gB, "xyz": gB["xyz"] @ Tba[:3, :3].T + Tba[:3, 3]}

    profA = wall_profile_pts(gA["xyz"])
    ip_gt = interpenetration(profA, gB_gt["xyz"])
    ip_ok = interpenetration(profA, gB_meas["xyz"])
    ip_fl = interpenetration(profA, gB_flip["xyz"])
    print(f"  interpenetration (B inside A):  GT {ip_gt*100:.0f}%   "
          f"measured-correct {ip_ok*100:.0f}%   flip {ip_fl*100:.0f}%")
    print(f"  => correct side {'coherent' if ip_ok<ip_fl else '??'}, "
          f"flip {'collides' if ip_fl>ip_ok else '??'} "
          f"(prior must pick the low-interpenetration side; embedding does, exp18 5/6)")

    out = config.RESULTS_ROOT / "gs"; out.mkdir(parents=True, exist_ok=True)
    gsi.write_gs_ply(out / f"e2_{sa[-6:]}_{sb[-6:]}_correct.ply", gsi_merge(gA, gB_meas))
    gsi.write_gs_ply(out / f"e2_{sa[-6:]}_{sb[-6:]}_flip.ply", gsi_merge(gA, gB_flip))

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    for x, (gB2, ttl, ip) in zip(ax, [(gB_gt, "GT pose", ip_gt),
                                      (gB_meas, "measured (correct side)", ip_ok),
                                      (gB_flip, "measured (FLIP — wrong side)", ip_fl)]):
        for g, c, lbl in [(gA, "#1f77b4", "A"), (gB2, "#d62728", "B")]:
            p = g["xyz"]; m = (p[:, 1] > -1.0) & (p[:, 1] < 0.8)
            idx = rng.choice(int(m.sum()), min(7000, int(m.sum())), replace=False)
            x.scatter(p[m][idx, 0], p[m][idx, 2], s=1, c=c, label=lbl)
        x.scatter([0], [0], c="k", marker="^", s=60)
        x.scatter([door[0]], [door[2]], c="orange", marker="*", s=120)
        x.set_aspect("equal"); x.grid(alpha=.3); x.legend(fontsize=8)
        x.set_title(f"{ttl}\nB-inside-A {ip*100:.0f}%", fontsize=10)
    fig.suptitle(f"E2: estimated-pose fusion + which-side flip — {sa[-10:]} + {sb[-10:]}  "
                 f"(orange * = door)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = out / f"e2_{sa[-6:]}_{sb[-6:]}_flip.png"; fig.savefig(p, dpi=110); print("  saved", p)


def gsi_merge(*gs):
    return {k: np.concatenate([g[k] for g in gs]) for k in gs[0]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=None); ap.add_argument("--b", default=None)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    main(a)

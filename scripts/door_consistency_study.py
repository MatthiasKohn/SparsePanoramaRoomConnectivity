"""Multi-floor study of the door-consistency (global-consistency) metric. GEOMETRY ONLY -> no GPU,
no rendering, no panos: runs on the ZInD jsons directly, so it scans many floors fast.

Answers two questions:
  (1) Is the lever-arm real? -> rotation-only vs translation-only noise. Rotation error is INVISIBLE
      to camera-centre RMSE yet should open large door gaps (delta swings a door ~room-radius away).
  (2) Independent vs coherent-drift noise: independent = every room errs on its own (pessimistic for
      neighbours); drift = error accumulates along the connectivity graph like a relative-pose
      estimator (neighbours stay locally consistent). door_gap measures NEIGHBOUR agreement, so drift
      should give much smaller gaps than independent at the same absolute RMSE.

    python scripts/door_consistency_study.py --zind <root> --n_floors 20 --out results/floor
"""
import argparse, glob, os, numpy as np, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sparsepano.datasets import zind_floor
from sparsepano.providers import poses as PRO_POSE, connectivity as PRO_CONN
from pipelines.floor import door_consistency, pose_errors


def eval_floor(js, floor, model, deg, m, seeds):
    """Return list of (pose_rmse_m, rot_err_deg, door_gap_m) over seeds for one floor/config."""
    fl = zind_floor.ZindFloor(Path(js), floor=floor)
    meters = float(fl.meters_per_coord)
    panos = [p for p in fl.panos if len(np.asarray(fl.panos[p]["verts_global"])) >= 3]
    rooms_map, adj = PRO_CONN.get_rooms(fl, panos, "gt")
    if len(rooms_map) < 3:
        return None
    out = []
    for sd in seeds:
        bp, rp, gp = PRO_POSE.get_poses(fl, meters, model, deg, m, seed=sd)
        d = door_consistency(fl, rooms_map, adj, bp, gp, meters)
        if d["n_doors"] == 0:
            return None
        r, ro = pose_errors(gp, bp, panos)
        out.append((r, ro, d["door_gap_m"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zind", required=True)
    ap.add_argument("--n_floors", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/floor")
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    os.makedirs(a.out, exist_ok=True)

    # collect floors that have >=3 rooms with shared doors
    jsons = sorted(glob.glob(os.path.join(a.zind, "*", "zind_data.json")))
    floors = []
    for js in jsons:
        for fk in ("floor_01", "floor_02", "floor_00"):
            try:
                if eval_floor(js, fk, "gt", 0, 0, [0]) is not None:
                    floors.append((js, fk))
            except Exception:
                pass
        if len(floors) >= a.n_floors:
            break
    floors = floors[: a.n_floors]
    print(f"using {len(floors)} floors")

    DEG = [0, 5, 10, 15, 20, 25]
    MPD = 0.02   # translation std = deg * this (couples the two like the sensitivity sweep)

    # ---- Q2: independent vs drift, aggregated across floors ----
    def curve(model):
        rms, gap = {d: [] for d in DEG}, {d: [] for d in DEG}
        for js, fk in floors:
            for d in DEG:
                res = eval_floor(js, fk, "gt" if d == 0 else model, d, round(d * MPD, 3), seeds)
                if res:
                    rms[d] += [x[0] for x in res]; gap[d] += [x[2] for x in res]
        return ([np.mean(rms[d]) for d in DEG],
                [np.mean(gap[d]) for d in DEG], [np.std(gap[d]) for d in DEG])
    xi, gi, gi_s = curve("noise")
    xd, gd, gd_s = curve("drift")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.errorbar(xi, gi, yerr=gi_s, fmt="o-", color="#a6161a", capsize=3, label="independent noise")
    ax.errorbar(xd, gd, yerr=gd_s, fmt="s--", color="#1f4e79", capsize=3, label="coherent drift")
    ax.plot([0, max(xi + xd)], [0, max(xi + xd)], ":", color="#888", lw=1, label="door gap = pose RMSE")
    ax.set_xlabel("absolute pose error (camera-centre RMSE, m)")
    ax.set_ylabel("door gap (m)")
    ax.set_title(f"Global consistency vs pose error  (mean over {len(floors)} ZInD floors)")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{a.out}/multifloor_independent_vs_drift.png", dpi=150)
    print("wrote multifloor_independent_vs_drift.png")

    # ---- Q1: lever arm -> rotation-only vs translation-only ----
    ROT = [0, 5, 10, 15, 20, 25]      # deg, translation off
    TRA = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # m, rotation off
    def levarm(kind):
        xs, gs = [], []
        vals = ROT if kind == "rot" else TRA
        for v in vals:
            rr, gg = [], []
            for js, fk in floors:
                deg = v if kind == "rot" else 0
                m = 0 if kind == "rot" else v
                res = eval_floor(js, fk, "gt" if v == 0 else "noise", deg, m, seeds)
                if res:
                    rr += [x[0] for x in res]; gg += [x[2] for x in res]
            xs.append(np.mean(rr)); gs.append(np.mean(gg))
        return xs, gs
    xr, gr = levarm("rot")
    xt, gt = levarm("tra")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot(gr, "o-", color="#a6161a", label="rotation-only (0–25°)")
    ax.plot(gt, "s--", color="#1f4e79", label="translation-only (0–0.5 m)")
    ax.set_xticks(range(len(ROT))); ax.set_xticklabels([f"{r}°\n{t}m" for r, t in zip(ROT, TRA)], fontsize=7)
    ax.set_xlabel("noise level (rotation° / translation m)")
    ax.set_ylabel("door gap (m)")
    ax.set_title(f"Lever arm: rotation vs translation error  (mean over {len(floors)} floors)")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{a.out}/leverarm_rot_vs_trans.png", dpi=150)
    print("wrote leverarm_rot_vs_trans.png")

    # dump a small csv
    with open(f"{a.out}/door_study.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "level", "pose_rmse_m", "door_gap_m"])
        for d, x, g in zip(DEG, xi, gi): w.writerow(["independent", d, round(x, 3), round(g, 3)])
        for d, x, g in zip(DEG, xd, gd): w.writerow(["drift", d, round(x, 3), round(g, 3)])
        for v, x, g in zip(ROT, xr, gr): w.writerow(["rot_only", v, round(x, 3), round(g, 3)])
        for v, x, g in zip(TRA, xt, gt): w.writerow(["trans_only", v, round(x, 3), round(g, 3)])
    print("\n== independent vs drift (mean door gap, m) ==")
    print(" deg  rmse  indep  drift")
    for d, x1, g1, x2, g2 in zip(DEG, xi, gi, xd, gd):
        print(f" {d:3d} {x1:5.3f}  {g1:5.3f}  {g2:5.3f}")
    print("\n== lever arm (mean door gap, m) ==")
    print(" lvl  rot-only  trans-only")
    for i, (r, t) in enumerate(zip(ROT, TRA)):
        print(f" {r:2d}°/{t:.1f}m  {gr[i]:6.3f}    {gt[i]:6.3f}")


if __name__ == "__main__":
    main()

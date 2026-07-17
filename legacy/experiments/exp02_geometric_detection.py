"""
exp02 — Geometric door detection without GT, then pose from detected apertures.

exp01 showed pose is well-posed GIVEN clean see-through points selected with GT.
This experiment removes that crutch: detect apertures geometrically from A's depth
alone, then recover pose. It measures the real cost of not having GT.

Three questions:
  (D) Detection: does a detected aperture cover the true see-through-toward-B
      region? (recall/precision vs geom.covisible_columns, the GT target)
  (P) Pose: V2 pose error using the residual-selected aperture vs exp01 oracle.
  (A) Association/connectivity: for the connected pair, is the lowest-residual
      aperture the one facing B? And for UNCONNECTED pairs, does the best
      residual stay high (so it can be rejected)?

Provider-flexible:  --source stanford | zind
Run:  python legacy/experiments/exp02_geometric_detection.py [--source stanford]
"""
import sys, os, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.geometry import geom
from sparsepano.doors import aperture
from sparsepano.pose import pose as posemod
from sparsepano.geometry import providers

PERTURB = (10.0, 0.4)   # yaw deg, translation m  (init for pose recovery)
N_TRIALS = 1
SEED = 0


def column_overlap(ap, gt_cols, W):
    """recall/precision of an aperture's azimuth span vs GT see-through columns."""
    span = np.zeros(W, bool)
    lo, hi = ap.u_lo, ap.u_hi
    if lo <= hi:
        span[lo:hi] = True
    else:
        span[lo:] = True; span[:hi] = True
    inter = np.sum(span & gt_cols)
    rec = inter / max(gt_cols.sum(), 1)
    prec = inter / max(span.sum(), 1)
    return rec, prec


def perturb_yaw(T_gt, yaw_deg, trans_m, rng):
    a = np.deg2rad(yaw_deg) * rng.choice([-1, 1])
    T = T_gt.copy()
    T[:3, :3] = T_gt[:3, :3] @ geom.Ry(a)
    dt = rng.normal(size=3); dt = dt / np.linalg.norm(dt) * trans_m
    T[:3, 3] = T_gt[:3, 3] + dt
    return T


def run(source="stanford"):
    rng = np.random.default_rng(SEED)
    if source == "zind":
        prov = providers.default_zind()
        if prov is None:
            print("[zind] no runnable ZInD data found; falling back to stanford")
            prov = providers.StanfordProvider()
    else:
        prov = providers.StanfordProvider()
    OUT = config.RESULTS_ROOT / f"exp02_geometric_detection_{prov.name}"
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = prov.pairs(max_connected=8, max_unconnected=8)
    print(f"Provider={prov.name}  pairs: {sum(p.connected for p in pairs)} connected, "
          f"{sum(not p.connected for p in pairs)} unconnected")

    # --- ZInD pose self-test: does GT rel_pose yield co-visible points? ---
    if prov.name == "zind":
        ok = 0; tested = 0
        for p in [x for x in pairs if x.connected][:5]:
            da, db = prov.depth(p.a), prov.depth(p.b)
            H, W = da.shape
            T = prov.rel_pose(p.a, p.b)
            cols, pts = geom.covisible_columns(da, db, T, W, H)
            tested += 1; ok += int(cols.sum() > 5)
        print(f"[zind self-test] co-visible found on {ok}/{tested} connected pairs "
              f"({'pose convention OK' if ok>=max(1,tested//2) else 'POSE CONVENTION SUSPECT'})")

    det_rows, pose_rows, ctrl_rows = [], [], []
    for p in pairs:
        hw = prov.hw(p.a)
        if hw is None:
            continue
        H, W = hw
        da, db = prov.depth(p.a), prov.depth(p.b)
        if da.shape != (H, W):
            import cv2; da = cv2.resize(da, (W, H))
        if db.shape != (H, W):
            import cv2; db = cv2.resize(db, (W, H))
        T_gt = prov.rel_pose(p.a, p.b)
        cands = aperture.detect_apertures(da)
        la, lb = prov.label(p.a), prov.label(p.b)

        if not cands:
            if p.connected:
                det_rows.append(dict(pair=f"{la}|{lb}", best_recall=0.0, best_prec=0.0,
                                     n_cands=0))
            continue

        # evaluate pose for each candidate; pick lowest converged residual
        results = []
        for c in cands:
            pts = aperture.points_from_mask(da, c.mask, rng=rng)
            if pts is None:
                continue
            best = None
            for _ in range(N_TRIALS):
                T_init = perturb_yaw(T_gt, *PERTURB, rng)
                out = posemod.recover(pts, db, T_init, W, H, variant="V2")
                if best is None or out["resid"] < best["resid"]:
                    best = out
            rot, trans = geom.pose_error(best["R"], best["t"], T_gt)
            results.append((c, best["resid"], rot, trans))
        if not results:
            continue
        results.sort(key=lambda x: x[1])     # by residual
        sel_c, sel_resid, sel_rot, sel_trans = results[0]

        if p.connected:
            gt_cols, _ = geom.covisible_columns(da, db, T_gt, W, H)
            # best detection over all candidates (detection quality, decoupled from pose)
            recs = [column_overlap(c, gt_cols, W) for c, *_ in results]
            best_rec = max(r for r, _ in recs); best_prec = max(pp for _, pp in recs)
            sel_rec, sel_prec = column_overlap(sel_c, gt_cols, W)
            det_rows.append(dict(pair=f"{la}|{lb}", best_recall=best_rec,
                                 best_prec=best_prec, n_cands=len(results)))
            pose_rows.append(dict(pair=f"{la}|{lb}", resid=sel_resid, rot=sel_rot,
                                  trans=sel_trans, sel_recall=sel_rec,
                                  n_cands=len(results)))
            print(f"  C {la:>12}|{lb:<12} cands={len(results)} "
                  f"detRecall={best_rec:.2f} selRot={sel_rot:5.1f}deg "
                  f"selTrans={sel_trans:4.2f}m resid={sel_resid:.3f}")
        else:
            ctrl_rows.append(dict(pair=f"{la}|{lb}", best_resid=sel_resid))
            print(f"  U {la:>12}|{lb:<12} cands={len(results)} bestResid={sel_resid:.3f}")

    _summary(det_rows, pose_rows, ctrl_rows, OUT, prov.name)


def _summary(det_rows, pose_rows, ctrl_rows, OUT, pname):
    import csv
    for fn, rows in [("detection.csv", det_rows), ("pose.csv", pose_rows),
                     ("control.csv", ctrl_rows)]:
        if rows:
            with open(OUT / fn, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader(); wr.writerows(rows)

    print("\n" + "=" * 60)
    if det_rows:
        rec = np.array([r["best_recall"] for r in det_rows])
        found = np.mean(rec > 0.3)
        print(f"DETECTION  : true doorway covered (recall>0.3) on {100*found:.0f}% "
              f"of connected pairs; median best recall={np.median(rec):.2f}")
    if pose_rows:
        rot = np.array([r["rot"] for r in pose_rows])
        trans = np.array([r["trans"] for r in pose_rows])
        good = np.array([r["sel_recall"] for r in pose_rows]) > 0.3
        print(f"POSE       : rot median={np.median(rot):.1f}deg (<5deg "
              f"{100*np.mean(rot<5):.0f}%), trans median={np.median(trans):.2f}m")
        print(f"             when selected aperture is the true one "
              f"({good.sum()}/{len(good)}): rot median={np.median(rot[good]) if good.any() else float('nan'):.1f}deg")
    if ctrl_rows and pose_rows:
        cr = np.array([r["best_resid"] for r in ctrl_rows])
        pr = np.array([r["resid"] for r in pose_rows])
        thr = (np.median(pr) + np.median(cr)) / 2
        print(f"CONTROL    : connected resid median={np.median(pr):.3f} vs "
              f"unconnected median={np.median(cr):.3f} "
              f"(separable: {'YES' if np.median(cr)>np.median(pr) else 'NO'})")

        plt.figure(figsize=(6, 4))
        plt.hist(pr, bins=12, alpha=0.6, label="connected (selected aperture)")
        plt.hist(cr, bins=12, alpha=0.6, label="unconnected (best aperture)")
        plt.xlabel("converged robust residual"); plt.ylabel("pairs"); plt.legend()
        plt.title(f"exp02 {pname}: residual separates connectivity")
        plt.tight_layout(); plt.savefig(OUT / "residual_separation.png", dpi=130)
    print("=" * 60)
    print(f"Wrote results to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="stanford", choices=["stanford", "zind"])
    a = ap.parse_args()
    run(a.source)

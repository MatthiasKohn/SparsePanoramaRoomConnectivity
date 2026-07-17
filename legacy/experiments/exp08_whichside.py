"""
exp08 — Resolve the door which-side flip with appearance (DINOv2), validated on ZInD.

Door-anchored pose (door_pose) leaves a binary ambiguity: the correct pose vs a
~180 deg flip, both with high geometric inlier. We break it with appearance:
A's view THROUGH the door shows part of B's room; under the CORRECT pose those
through-door points project to where B actually sees that content, so a crop of B
there should look like A's through-door crop. Under the flip they project onto the
wrong wall -> lower similarity. Pick the higher-similarity candidate.

Because ZInD has GT pose, we can MEASURE whether appearance picks the right side.

    pip install torch torchvision pillow            # DINOv2 via torch.hub
    python legacy/experiments/exp08_whichside.py
    python legacy/experiments/exp08_whichside.py --selftest # no model, checks plumbing
"""
import sys, os, argparse
import numpy as np, cv2

from sparsepano import config
from sparsepano.geometry import geom
from sparsepano.geometry import providers
from sparsepano.datasets import zind
from sparsepano.pose import door_pose
from sparsepano.geometry import panoproj
from sparsepano.pose import pose as posemod
from sparsepano.pose.matching import DoorMatcher


def candidate(da, db, aza, azb, yaw0_deg, W, H, rng):
    """Refine a pose around a given yaw init; return (pose, through-door points)."""
    P, d_a = door_pose.through_door_points(da, aza, W, H, rng=rng)
    if P is None:
        return None
    _, d_b = door_pose._sector_wall(db, azb, W, H)
    camb = (d_a + d_b) * door_pose._dir(aza)
    best = None
    for dy in np.radians(np.linspace(-15, 15, 5)):
        R0 = geom.Ry(np.radians(yaw0_deg) + dy)
        Ti = np.eye(4); Ti[:3, :3] = R0; Ti[:3, 3] = -R0 @ camb
        o = posemod.recover(P, db, Ti, W, H, variant="V2", max_nfev=110)
        res, _ = geom.residual(P, db, o["R"], o["t"], o["s"], W, H)
        inl = float((np.abs(res) < 0.2).mean()) if res.size else 0.0
        if best is None or inl > best[0]:
            best = (inl, o)
    return best[1], P


def proj_azimuth(P, pose):
    pb = P @ pose["R"].T + pose["t"]
    az = np.degrees(np.arctan2(pb[:, 0], pb[:, 2]))
    return float(np.median(az))


def rgb_zind(stem):
    p = config.zind_paths()["panos"] / f"{stem}.jpg"
    im = cv2.imread(str(p))
    return cv2.cvtColor(cv2.resize(im, (4096, 2048)), cv2.COLOR_BGR2RGB) if im is not None else None


def main(selftest=False):
    prov = providers.default_zind()
    matcher = DoorMatcher(embed=(lambda img: np.array([1.0, 0.0])) if selftest else None,
                          fov_deg=60.0, crop_hw=(448, 448))
    conn = [p for p in prov.pairs(max_connected=30, max_unconnected=0) if p.connected]
    rng = np.random.default_rng(0)
    n_ok, n_tot, pose_err = 0, 0, []
    print(f"{'pair':10} {'pick':>6} {'simC':>5} {'simF':>5} {'rotErr':>6} {'correct?':>8}")
    for p in conn:
        ba = prov.shared_door_bearing(p.a, p.b); bb = prov.shared_door_bearing(p.b, p.a)
        if ba is None or bb is None:
            continue
        rgbA, rgbB = rgb_zind(p.a), rgb_zind(p.b)
        if rgbA is None or rgbB is None:
            continue
        da, db = prov.depth(p.a), prov.depth(p.b); H, W = da.shape
        Tg = prov.rel_pose(p.a, p.b)
        aza, azb = np.degrees(ba), np.degrees(bb)
        cands = []
        for tag, yaw0 in [("correct", azb - aza + 180), ("flip", azb - aza)]:
            c = candidate(da, db, aza, azb, yaw0, W, H, rng)
            if c:
                cands.append((tag, *c))
        if len(cands) < 2:
            continue
        Acrop = panoproj.e2p(rgbA, aza, 0, 60, (448, 448))
        eA = matcher._embed(Acrop)
        scored = []
        for tag, pose, P in cands:
            Bcrop = panoproj.e2p(rgbB, proj_azimuth(P, pose), 0, 60, (448, 448))
            sim = float(eA @ matcher._embed(Bcrop))
            scored.append((sim, tag, pose))
        scored.sort(reverse=True)
        pick = scored[0]
        re, _ = geom.pose_error(pick[2]["R"], pick[2]["t"], Tg)
        correct = re < 30
        n_tot += 1; n_ok += int(correct); pose_err.append(re)
        sims = {t: s for s, t, _ in scored}
        print(f"{p.a.split('room_')[1][:8]:10} {pick[1]:>6} {sims.get('correct',0):5.2f} "
              f"{sims.get('flip',0):5.2f} {re:6.1f} {str(correct):>8}")
    if n_tot:
        print(f"\nappearance picked correct side: {n_ok}/{n_tot} ({100*n_ok/n_tot:.0f}%)  "
              f"median rotErr after disambig = {np.median(pose_err):.1f} deg")
    else:
        print("no runnable pairs (need ZInD RGB + depth + shared door)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    main(ap.parse_args().selftest)

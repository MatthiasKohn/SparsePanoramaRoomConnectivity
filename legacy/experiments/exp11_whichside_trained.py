"""
exp11 — Which-side disambiguation with the TRAINED contrastive encoder (vs the 60%
off-the-shelf DINOv2 baseline from exp08). Validated on ZInD with GT.

Same protocol as exp08: build the two candidate poses (correct-init vs flip), embed
A's through-door view and B's content where each candidate projects, pick the higher
similarity, score the picked pose against GT. Only the embedder changes.

  python legacy/experiments/exp11_whichside_trained.py --ckpt door_encoder.pt
"""
import sys, os, argparse
import numpy as np, cv2

from sparsepano import config
from sparsepano.geometry import geom
from sparsepano.geometry import providers
from sparsepano.pose import door_pose
from sparsepano.geometry import panoproj
from sparsepano.pose import pose as posemod
from sparsepano.doors import contrastive


def candidate(da, db, aza, azb, yaw0_deg, W, H, rng):
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


def proj_az(P, pose):
    pb = P @ pose["R"].T + pose["t"]
    return float(np.median(np.degrees(np.arctan2(pb[:, 0], pb[:, 2]))))


def rgb_zind(stem):
    p = config.zind_paths()["panos"] / f"{stem}.jpg"
    im = cv2.imread(str(p))
    return cv2.cvtColor(cv2.resize(im, (4096, 2048)), cv2.COLOR_BGR2RGB) if im is not None else None


def main(ckpt):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    embed = contrastive.load_embedder(ckpt, dev)
    prov = providers.default_zind()
    conn = [p for p in prov.pairs(max_connected=30, max_unconnected=0) if p.connected]
    rng = np.random.default_rng(0)
    nok = nt = 0; errs = []
    print(f"{'pair':10} {'pick':>7} {'simC':>5} {'simF':>5} {'rotErr':>6} {'ok?':>5}")
    for p in conn:
        ba = prov.shared_door_bearing(p.a, p.b); bb = prov.shared_door_bearing(p.b, p.a)
        if ba is None or bb is None:
            continue
        rA, rB = rgb_zind(p.a), rgb_zind(p.b)
        if rA is None or rB is None:
            continue
        da, db = prov.depth(p.a), prov.depth(p.b); H, W = da.shape
        Tg = prov.rel_pose(p.a, p.b); aza, azb = np.degrees(ba), np.degrees(bb)
        cands = []
        for tag, y0 in [("correct", azb - aza + 180), ("flip", azb - aza)]:
            c = candidate(da, db, aza, azb, y0, W, H, rng)
            if c:
                cands.append((tag, *c))
        if len(cands) < 2:
            continue
        eA = embed(panoproj.e2p(rA, aza, 0, 60, (224, 224)))
        eA = eA / (np.linalg.norm(eA) + 1e-8)
        scored = []
        for tag, pose, P in cands:
            eB = embed(panoproj.e2p(rB, proj_az(P, pose), 0, 60, (224, 224)))
            eB = eB / (np.linalg.norm(eB) + 1e-8)
            scored.append((float(eA @ eB), tag, pose))
        scored.sort(reverse=True)
        re, _ = geom.pose_error(scored[0][2]["R"], scored[0][2]["t"], Tg)
        ok = re < 30; nok += ok; nt += 1; errs.append(re)
        sims = {t: s for s, t, _ in scored}
        print(f"{p.a.split('room_')[1][:8]:10} {scored[0][1]:>7} {sims.get('correct',0):5.2f} "
              f"{sims.get('flip',0):5.2f} {re:6.1f} {str(ok):>5}")
    if nt:
        print(f"\nTRAINED encoder: correct side {nok}/{nt} ({100*nok/nt:.0f}%)  "
              f"median rotErr={np.median(errs):.1f} deg   (exp08 DINOv2 baseline was 60%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="door_encoder.pt")
    main(ap.parse_args().ckpt)

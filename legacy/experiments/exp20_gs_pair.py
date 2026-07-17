"""
exp20 (E1) — Two-room through-door Gaussian fusion (GT pose).

Fuses two connected rooms' per-pano Gaussian inits (exp19) into ONE frame via the GT
relative pose, isolating cross-room STITCHING from pose error. CPU sanity:
  - top-down of the fused cloud (the two rooms should sit ADJACENT, sharing the doorway,
    not interpenetrating) -> validates the placement;
  - a novel view shifted toward the doorway, A-only vs fused: room B should FILL the
    through-door disocclusion holes that A alone cannot (the cross-room payoff).
Exports a fused 3DGS .ply. The joint differentiable GS optimization (render against BOTH
panos, refine geometry+pose) is isolated in `optimize_pair_gs` -> run on GPU.

  python legacy/experiments/exp20_gs_pair.py                     # auto-pick a connected pair
  python legacy/experiments/exp20_gs_pair.py --a <stemA> --b <stemB>
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


def merge(*gs):
    return {k: np.concatenate([g[k] for g in gs]) for k in gs[0]}


def transform(g, R, t):
    out = dict(g); out["xyz"] = g["xyz"] @ R.T + t
    return out


def optimize_pair_gs(ply_path, panoA, panoB, T_ba, iters=5000, device="cuda"):
    """HOOK (GPU): joint differentiable GS optimization rendering against BOTH panos
    (A at identity, B at pose T_ba), refining Gaussians (and optionally the pose). Now
    there are two viewpoints, so disoccluded regions get real supervision. Needs a
    Gaussian rasterizer (gsplat / diff-gaussian-rasterization)."""
    raise NotImplementedError("joint GS optimization: run on a GPU machine with a rasterizer")


def auto_pair(prov):
    names = [n for n in prov.fl.panos if (prov.depth_dir / f"{n}.npy").exists()]
    best = None
    for a, b in combinations(names, 2):
        if prov.fl.panos[a]["room"] == prov.fl.panos[b]["room"]:
            continue
        if prov.fl.shared_door(a, b) is None:
            continue
        az = prov.shared_door_bearing(a, b)
        if az is None:
            continue
        da = prov.depth(a); H, W = da.shape
        inl = door_pose.recover(da, prov.depth(b), np.degrees(az),
                                np.degrees(prov.shared_door_bearing(b, a)), W, H)
        sc = inl["inlier"] if inl else 0
        if best is None or sc > best[2]:
            best = (a, b, sc)
    return best[0], best[1]


def main(a):
    prov = providers.default_zind()
    H, W = a.res, a.res * 2
    sa, sb = (a.a, a.b) if a.a and a.b else auto_pair(prov)
    print(f"pair: {sa}  <->  {sb}")

    dA = cv2.resize(prov.depth(sa), (W, H), interpolation=cv2.INTER_NEAREST)
    dB = cv2.resize(prov.depth(sb), (W, H), interpolation=cv2.INTER_NEAREST)
    gA = gsi.gaussian_init_from_pano(dA, load_pano(prov, sa, (H, W)), stride=a.stride, scale_mult=a.gscale)
    gB = gsi.gaussian_init_from_pano(dB, load_pano(prov, sb, (H, W)), stride=a.stride, scale_mult=a.gscale)

    # place B in A's frame via GT pose (camB -> camA)
    Tba = prov.rel_pose(sb, sa); Rba, tba = Tba[:3, :3], Tba[:3, 3]
    gB_A = transform(gB, Rba, tba)
    fused = merge(gA, gB_A)
    print(f"  gaussians: A {len(gA['xyz']):,}  B {len(gB['xyz']):,}  fused {len(fused['xyz']):,}")

    # doorway viewpoint + novel offset toward the door (in A frame)
    az_a = prov.shared_door_bearing(sa, sb)
    ddir = np.array([np.sin(az_a), 0.0, np.cos(az_a)])
    tcam = a.baseline * ddir                                  # move A toward the door
    novA, mA = gsi.render_equirect(gA, H, W, t=-tcam)
    novF, mF = gsi.render_equirect(fused, H, W, t=-tcam)
    cov_a, cov_f = float(mA.mean()), float(mF.mean())
    print(f"  novel view (+{a.baseline} m toward door): coverage A-only {cov_a*100:.0f}%  "
          f"fused {cov_f*100:.0f}%  -> B fills {(cov_f-cov_a)*100:.0f}% more")

    out = config.RESULTS_ROOT / "gs"; out.mkdir(parents=True, exist_ok=True)
    gsi.write_gs_ply(out / f"pair_{sa[-6:]}_{sb[-6:]}_gs.ply", fused)
    gsi.write_point_ply(out / f"pair_{sa[-6:]}_{sb[-6:]}_points.ply", fused)

    fig = plt.figure(figsize=(16, 5)); gsr = fig.add_gridspec(1, 3)
    # top-down adjacency
    ax0 = fig.add_subplot(gsr[0, 0])
    for g, c, lbl in [(gA, "#1f77b4", "room A"), (gB_A, "#d62728", "room B")]:
        p = g["xyz"]; m = (p[:, 1] > -1.0) & (p[:, 1] < 0.8)
        s = np.random.default_rng(0).choice(int(m.sum()), min(8000, int(m.sum())), replace=False)
        ax0.scatter(p[m][s, 0], p[m][s, 2], s=1, c=c, label=lbl)
    ax0.scatter([0], [0], c="k", marker="^", s=60); ax0.set_aspect("equal")
    ax0.legend(fontsize=8); ax0.set_title("top-down fused (rooms adjacent via doorway)", fontsize=10)
    ax0.grid(alpha=.3)
    a1 = fig.add_subplot(gsr[0, 1]); a1.imshow(novA)
    a1.set_title(f"novel view, A only (holes, {cov_a*100:.0f}%)", fontsize=10); a1.axis("off")
    a2 = fig.add_subplot(gsr[0, 2]); a2.imshow(novF)
    a2.set_title(f"novel view, fused (B fills through-door, {cov_f*100:.0f}%)", fontsize=10); a2.axis("off")
    fig.suptitle(f"E1: two-room through-door fusion (GT pose) — {sa[-10:]} + {sb[-10:]}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out / f"pair_{sa[-6:]}_{sb[-6:]}_fusion.png"; fig.savefig(p, dpi=110); print("  saved", p)

    if a.optimize:
        from sparsepano.gs import gs_optim
        panos = [load_pano(prov, sa, (H, W)), load_pano(prov, sb, (H, W))]
        poses = [np.eye(4, dtype=np.float32), Tba.astype(np.float32)]   # A=world, B=camB->camA
        gs_optim.convention_check(fused, panos, poses, opengl=a.opengl, device=a.device)
        if a.debug_tiles:
            gs_optim.save_debug_tiles(fused, panos, poses, out / f"pair_{sa[-6:]}_{sb[-6:]}_tiles.png",
                                      device=a.device)
        g2 = gs_optim.optimize(fused, panos, poses, iters=a.iters, opengl=a.opengl, device=a.device)
        gsi.write_gs_ply(out / f"pair_{sa[-6:]}_{sb[-6:]}_opt.ply", g2)
        gsi.write_point_ply(out / f"pair_{sa[-6:]}_{sb[-6:]}_opt_points.ply", g2)   # colored, Open3D
        print("  wrote optimized", out / f"pair_{sa[-6:]}_{sb[-6:]}_opt.ply", "(+ _points.ply)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=None); ap.add_argument("--b", default=None)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--baseline", type=float, default=0.8)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--opengl", action="store_true", help="(unused; convention auto-detected)")
    ap.add_argument("--debug_tiles", action="store_true", help="dump target|render tiles + per-pano PSNR")
    ap.add_argument("--gscale", type=float, default=1.5, help="init Gaussian scale multiplier (larger = more opaque walls, less bleed)")
    a = ap.parse_args()
    main(a)

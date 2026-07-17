"""
exp16 — Photometric (rendering) consistency as a pose signal — the seed of the
3D-reconstruction / DepthSplat-style direction, testable on ZInD with GT.

Forward model (what a Gaussian-Splatting reconstruction optimizes, in miniature):
warp pano A's colored points into pano B's view via a candidate relative pose, then
compare the warped colors to B's real image. The GEOMETRIC residual only checks
range (it was ambiguous — collapse & which-side flip). The PHOTOMETRIC residual
checks COLOR of the shared (through-door) surfaces, which should pin the true pose.

We sweep yaw around the GT pose and plot both signals. Expectation:
  - geometric inlier: multi-modal / flat (ambiguous);
  - photometric error: a single sharp minimum at the true pose.
If so, a rendering loss is exactly what resolves the flip (and motivates GS pose
refinement next).

  python legacy/experiments/exp16_photometric_pose.py
"""
import sys, os
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.geometry import geom
from sparsepano.geometry import providers


def load(prov, stem, hw=(1024, 2048)):
    d = cv2.resize(prov.depth(stem), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    im = cv2.imread(str(config.zind_paths()["panos"] / f"{stem}.jpg"))
    rgb = cv2.cvtColor(cv2.resize(im, (hw[1], hw[0])), cv2.COLOR_BGR2RGB).astype(np.float32)
    return d, rgb


def warp(ptsA, R, t, W, H):
    pb = ptsA @ R.T + t
    u, v, r = geom.project_to_pano(pb, W, H)
    return u, v, r


def photo_and_geom(ptsA, cA, depthB, rgbB, R, t, W, H):
    """Over a FIXED point set: photometric colour error and geometric range error."""
    u, v, r = warp(ptsA, R, t, W, H)
    meas = geom.sample_bilinear(depthB, u, v)
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    cB = rgbB[vi, ui]
    valid = np.isfinite(meas) & (meas > 0.1)
    if valid.sum() < 50:
        return np.nan, np.nan
    a = cA[valid]; b = cB[valid]
    photo_l1 = float(np.mean(np.abs(a - b)) / 255.0)
    ga = a @ [0.299, 0.587, 0.114]; gb = b @ [0.299, 0.587, 0.114]   # grayscale
    ncc = float(np.corrcoef(ga, gb)[0, 1]) if ga.std() > 1e-6 and gb.std() > 1e-6 else 0.0
    return photo_l1, 1.0 - ncc        # exposure/gain-invariant structural error


def main():
    prov = providers.default_zind()
    pairs = [p for p in prov.pairs(max_connected=6, max_unconnected=0) if p.connected][:3]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    degs = np.arange(-180, 181, 5)
    for ax, p in zip(axes, pairs):
        dA, rgbA = load(prov, p.a); dB, rgbB = load(prov, p.b); H, W = dA.shape
        Tg = prov.rel_pose(p.a, p.b); Rg, tg = Tg[:3, :3], Tg[:3, 3]
        ptsA, us, vs = geom.backproject(dA, stride=2)
        cAall = rgbA[vs.astype(int), us.astype(int)]
        # FIXED shared set: A-points that are range-consistent at the GT pose
        u0, v0, r0 = warp(ptsA, Rg, tg, W, H)
        meas0 = geom.sample_bilinear(dB, u0, v0)
        shared = np.isfinite(meas0) & (meas0 > 0.1) & (np.abs(r0 - meas0) < 0.25)
        Ps, cs = ptsA[shared], cAall[shared]
        if len(Ps) > 12000:
            idx = rng.choice(len(Ps), 12000, replace=False); Ps, cs = Ps[idx], cs[idx]
        l1, nccs = [], []
        for dlt in degs:
            R = Rg @ geom.Ry(np.radians(dlt))
            ph, nc = photo_and_geom(Ps, cs, dB, rgbB, R, tg, W, H)
            l1.append(ph); nccs.append(nc)
        l1 = np.array(l1); nccs = np.array(nccs)
        ax.plot(degs, l1 / (np.nanmax(l1) + 1e-9), label="raw-RGB L1", color="#ff7f0e")
        ax.plot(degs, nccs / (np.nanmax(nccs) + 1e-9), label="1-NCC (exposure-inv)", color="#d62728")
        ax.axvline(0, color="k", lw=0.8); ax.axvline(180, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{p.a.split('room_')[1][:6]}<->{p.b.split('room_')[1][:6]}  "
                     f"(n={len(Ps)})", fontsize=9)
        ax.set_xlabel("yaw offset from GT (deg)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        print(f"{p.a.split('room_')[1][:8]}<->{p.b.split('room_')[1][:8]}: "
              f"raw-L1 min yaw={degs[np.nanargmin(l1)]:+d}, NCC min yaw={degs[np.nanargmin(nccs)]:+d} (GT=0)")
    fig.suptitle("Photometric vs geometric pose landscape (ZInD, yaw sweep around GT)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = config.RESULTS_ROOT / "photometric"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "yaw_landscape.png", dpi=120); print("saved", out / "yaw_landscape.png")


if __name__ == "__main__":
    main()

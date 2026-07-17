"""
exp04 — Proof of concept on real partner data (Immersight), NO ground truth.

One floor, 19 high-res panoramas, big overlap, depths from DAP. We show the whole
pipeline behaving end-to-end and produce a single explainable figure:

  (a) RGB panorama (the input)
  (b) monocular depth (DAP)
  (c) top-down room shape from that single depth + camera + recovered direction to B
  (d) two panoramas registered with NO pose prior (multi-start yaw + free-scale V1),
      both camera positions shown -> a coherent local floor map.

Notes / honest caveats baked into the method here:
  * DAP depth is affine (per-image scale). With big overlap we therefore free the
    scale (V1); the overlap constrains it. (Sparse-doorway data uses fixed-scale V2.)
  * The global metric scale is unknown without GT; a cosmetic uniform VIS factor only
    makes the plot room-sized. Relative geometry is what the method recovers.

Run:  python legacy/experiments/exp04_immersight.py
"""
import sys, os
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sparsepano import config
from sparsepano.geometry import geom
from sparsepano.pose import pose as posemod

IMM = config.DATA_ROOT / "Download_immersight_ 2026-06-23_10-23-49"
DD = IMM / "dap_depth" / "depth_meters"
OUT = config.RESULTS_ROOT / "exp04_immersight"; OUT.mkdir(parents=True, exist_ok=True)
VIS = 2.5            # cosmetic uniform scale (applied to BOTH panos) so plot is room-sized
ANCHOR = "panorama_1273532"
NEIGHBOR = "panorama_1273535"


def rgb_path(stem):
    for ext in (".jpeg", ".png", ".jpg"):
        p = IMM / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_depth(stem, h=1024):
    d = np.load(DD / f"{stem}.npy").astype(np.float32)
    return cv2.resize(d, (h * 2, h), interpolation=cv2.INTER_NEAREST)


def inlier(pts, db, R, t, s, W, H, tau=0.12):
    res, val = geom.residual(pts, db, R, t, s, W, H)
    return (float((np.abs(res) < tau).mean()) if res.size else 0.0,
            float(val.mean()))


def register(pts_a, db, W, H, variant="V1", yaw_steps=24, tau=0.12):
    best = None
    for y in np.linspace(0, 2 * np.pi, yaw_steps, endpoint=False):
        Ti = np.eye(4); Ti[:3, :3] = geom.Ry(y)
        o = posemod.recover(pts_a, db, Ti, W, H, variant=variant, max_nfev=140)
        f, c = inlier(pts_a, db, o["R"], o["t"], o["s"], W, H, tau)
        if best is None or f * c > best[0]:
            best = (f * c, o, f, y)
    return best[1], best[2], best[3]


def bev(P, ylo=-0.9, yhi=0.6):
    P = P[(P[:, 1] > ylo) & (P[:, 1] < yhi)]
    return P[:, 0] * VIS, P[:, 2] * VIS


def main():
    rng = np.random.default_rng(0)
    da = load_depth(ANCHOR); db = load_depth(NEIGHBOR); H, W = da.shape
    pa, _, _ = geom.backproject(da, stride=2)
    pa = pa[rng.choice(len(pa), 9000, replace=False)]
    o, inl, yaw = register(pa, db, W, H, variant="V1")
    R, t, s = o["R"], o["t"], o["s"]
    camB = (-t) @ R * VIS
    print(f"registered {ANCHOR}+{NEIGHBOR}: yaw={np.degrees(yaw):.0f} "
          f"inlier={inl:.2f} scale={s:.2f} baseline={np.linalg.norm(t)*VIS:.2f}m(vis)")

    rgb = cv2.cvtColor(cv2.imread(str(rgb_path(ANCHOR))), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (1100, 550))
    dvis = da.copy(); lo, hi = np.percentile(dvis, [2, 98])
    dcol = cm.get_cmap("turbo")(np.clip((dvis - lo) / (hi - lo), 0, 1))[..., :3]

    PA, _, _ = geom.backproject(da, stride=3)
    PB, _, _ = geom.backproject(db, stride=3)
    ax_, az_ = bev(PA); bx_, bz_ = bev((PB - t) @ R / s)
    sax, saz = bev(PA)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(2, 2, 1); ax.imshow(rgb); ax.axis("off")
    ax.set_title("(a) input panorama  ·  Immersight " + ANCHOR[-7:], fontsize=11)

    ax = fig.add_subplot(2, 2, 2); ax.imshow(dcol); ax.axis("off")
    ax.set_title("(b) monocular depth (DAP)  ·  near→far", fontsize=11)

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(sax, saz, s=1, c="#1f77b4", alpha=.3)
    ax.scatter([0], [0], c="k", marker="*", s=140, zorder=5)
    ang = np.arctan2(camB[0], camB[2])
    ax.annotate("", xy=(3.5*np.sin(ang), 3.5*np.cos(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2))
    ax.text(0.02, 0.98, "→ recovered direction\n   to neighbour", color="#d62728",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.grid(alpha=.2)
    ax.set_title("(c) top-down room shape from one depth", fontsize=11)
    ax.set_xlabel("metres (relative)")

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(ax_, az_, s=1, c="#1f77b4", alpha=.25, label=ANCHOR[-7:])
    ax.scatter(bx_, bz_, s=1, c="#d62728", alpha=.25, label=NEIGHBOR[-7:] + " → A")
    ax.scatter([0], [0], c="#1f77b4", marker="*", s=160, ec="k", zorder=6)
    ax.scatter([camB[0]], [camB[2]], c="#d62728", marker="*", s=160, ec="k", zorder=6)
    ax.set_aspect("equal"); ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.grid(alpha=.2)
    ax.legend(markerscale=8, loc="lower left")
    ax.set_title(f"(d) two panos registered, NO pose prior\n"
                 f"yaw {np.degrees(yaw):.0f}°, rel-scale {s:.2f}, inlier {inl:.2f}", fontsize=11)
    ax.set_xlabel("metres (relative)")

    fig.suptitle("Doorway/overlap pose recovery on real partner data (Immersight, no GT)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "immersight_poc.png", dpi=120)
    print("saved", OUT / "immersight_poc.png")


if __name__ == "__main__":
    main()

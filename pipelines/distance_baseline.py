"""
exp31 (M2 baseline) — camera->door DISTANCE: DAP depth vs door-geometry vs GT.

The PaperV2 metric-pose claim is "drop monocular depth; get camera->door distance from the
door itself." Before building a learned head, measure the bar on depth-equipped ZInD floors:

  per door, three estimates of the camera->door distance:
    GT           : |door_mid_global - camera| * meters_per_coord      (from data_floors geometry)
    DAP          : monocular-depth value sampled in the door's bearing sector (the exp27 path)
    fixed-width  : W0 / (2 tan(angular_width/2)),  angular_width from the door's two endpoint
                   bearings, W0 = a FIXED door-width prior (no per-door GT) -> a depth-free estimate

Reports MAE/median error of DAP and fixed-width vs GT. If DAP's door-distance error is large
(the ~2 m layout bottleneck) and the depth-free geometric estimate competes, the "drop depth"
thesis is validated -> the learned distance head only has to match/beat this.

  python -m pipelines.distance_baseline --home ../data/zind/full_dataset/0025 \
      --depth_dir ../data/zind/full_dataset/0025/dap_depth/depth_meters --floor floor_01
"""
import sys, os, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.datasets import zind
from sparsepano.doors import door_dataset


def wrap180(a):
    return (a + 180) % 360 - 180


def sample_depth_sector(depth, az_deg, half_deg):
    """Median depth in the horizon band within +/- half_deg of azimuth az (radial distance)."""
    H, W = depth.shape
    u0 = ((az_deg / 360.0) + 0.5) * W
    du = max(1, int(half_deg / 360.0 * W))
    cols = (np.arange(int(u0) - du, int(u0) + du + 1) % W)
    band = depth[int(H * 0.45):int(H * 0.55), :][:, cols]
    band = band[np.isfinite(band) & (band > 0.1)]
    return float(np.median(band)) if band.size else np.nan


def main(a):
    jp = Path(a.home) / "zind_data.json"
    fl = zind.ZindFloor(jp, floor=a.floor)
    S = fl.meters_per_coord
    ddir = Path(a.depth_dir)

    gt, dap, fw, awid = [], [], [], []
    depth_cache = {}
    for pano, info in fl.panos.items():
        dp = ddir / f"{pano}.npy"
        if not dp.exists():
            continue
        if pano not in depth_cache:
            import cv2
            d = np.load(dp).astype(np.float32)
            depth_cache[pano] = d
        depth = depth_cache[pano]
        cam = np.array(info["pos"], float)
        for (d0, d1) in info["doors_global"]:
            mid = (d0 + d1) / 2.0
            b_mid = door_dataset.door_azimuth(fl, pano, mid)
            b0 = door_dataset.door_azimuth(fl, pano, d0)
            b1 = door_dataset.door_azimuth(fl, pano, d1)
            ang_w = abs(wrap180(b0 - b1))                      # angular width (deg)
            if ang_w < 0.5 or ang_w > 120:                     # skip degenerate/behind-camera
                continue
            g = float(np.linalg.norm(mid - cam) * S)
            dd = sample_depth_sector(depth, b_mid, ang_w / 2)
            fwd = a.w0 / (2 * np.tan(np.radians(ang_w) / 2))
            if not np.isfinite(dd):
                continue
            gt.append(g); dap.append(dd); fw.append(fwd); awid.append(ang_w)

    gt, dap, fw = np.array(gt), np.array(dap), np.array(fw)
    n = len(gt)
    def err(x): return np.abs(x - gt)
    print(f"home {Path(a.home).name} {a.floor}: {n} doors with depth")
    print(f"  camera->door distance error vs GT (median | MAE):")
    print(f"    DAP depth       : {np.median(err(dap)):.2f} | {err(dap).mean():.2f} m")
    print(f"    fixed-width({a.w0:.2f}m): {np.median(err(fw)):.2f} | {err(fw).mean():.2f} m")
    print(f"  GT distance range {gt.min():.2f}-{gt.max():.2f} m (median {np.median(gt):.2f})")
    winner = "fixed-width geometry" if np.median(err(fw)) < np.median(err(dap)) else "DAP depth"
    print(f"  => lower-error estimator: {winner}")

    out = config.RESULTS_ROOT / "distance"; out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].scatter(gt, dap, s=12, c="#d62728", label=f"DAP (med err {np.median(err(dap)):.2f} m)")
    ax[0].scatter(gt, fw, s=12, c="#1f77b4", label=f"fixed-width (med err {np.median(err(fw)):.2f} m)")
    lim = [0, max(gt.max(), dap.max(), fw.max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("GT distance (m)"); ax[0].set_ylabel("estimated (m)"); ax[0].legend(fontsize=8)
    ax[0].set_title("camera->door distance"); ax[0].set_aspect("equal")
    ax[1].hist(err(dap), bins=20, alpha=.6, color="#d62728", label="DAP")
    ax[1].hist(err(fw), bins=20, alpha=.6, color="#1f77b4", label="fixed-width")
    ax[1].set_xlabel("abs error vs GT (m)"); ax[1].legend(fontsize=8); ax[1].set_title("error distribution")
    fig.suptitle(f"M2 distance baseline — {Path(a.home).name} ({n} doors)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out / f"distance_{Path(a.home).name}.png"; fig.savefig(p, dpi=120); print("  saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--w0", type=float, default=0.9, help="fixed door-width prior (m)")
    a = ap.parse_args()
    main(a)

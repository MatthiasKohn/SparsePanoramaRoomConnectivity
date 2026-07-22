"""
depth_ab — A/B camera->door DISTANCE error for two (or more) depth sources over many homes.

THE decision metric for adopting PaGeR as the geometry backbone. ZInD has no GT depth, but it HAS
GT door geometry, so we score each depth source by how well its depth-derived camera->door distance
matches the true distance. Lower error = better depth *for our task*. Compares on the PAIRED set of
doors where every source has a depth value (fair A/B), plus the depth-free fixed-width geometry baseline.

  python -m pipelines.depth_ab --root $ZIND_ROOT --homes scripts/depth_homes.txt \
      --sources pager_depth/depth_meters,dap_depth/depth_meters

Runs on CPU in seconds — no GPU needed (login node or a short srun is fine).
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.datasets import zind
from sparsepano.doors import door_dataset
from pipelines.distance_baseline import wrap180, sample_depth_sector


_warned = set()


def _load_depth_2d(path):
    """Load a depth .npy and coerce to (H,W). PaGeR saves (1,H,W)/(H,W,1) etc; squeeze/reduce it.
    Returns None (with a one-time warning) if it can't be reduced to a single-channel map."""
    d = np.load(path).astype(np.float32)
    d = np.squeeze(d)
    if d.ndim == 3:
        if d.shape[0] == 1:          # (1,H,W)
            d = d[0]
        elif d.shape[-1] == 1:       # (H,W,1)
            d = d[..., 0]
    if d.ndim != 2:
        key = str(path.parent)
        if key not in _warned:
            _warned.add(key)
            print(f"  [warn] {path.parent.parent.name}/{path.parent.name}: depth shape {np.load(path).shape} "
                  f"is not a single-channel map (squeezed -> {d.shape}); this source will be skipped for it.")
        return None
    return d


def _floors(home):
    try:
        return list(json.load(open(home / "zind_data.json")).get("merger", {}).keys())
    except Exception:
        return []


def _home_rows(home, floor, sources, w0):
    try:
        fl = zind.ZindFloor(home / "zind_data.json", floor=floor)
    except Exception:
        return []
    S = fl.meters_per_coord
    rows = []
    for pano, info in fl.panos.items():
        depths = {}
        for s in sources:
            dp = home / s / f"{pano}.npy"
            depths[s] = _load_depth_2d(dp) if dp.exists() else None
        cam = np.array(info["pos"], float)
        for (d0, d1) in info["doors_global"]:
            mid = (d0 + d1) / 2.0
            b_mid = door_dataset.door_azimuth(fl, pano, mid)
            b0 = door_dataset.door_azimuth(fl, pano, d0)
            b1 = door_dataset.door_azimuth(fl, pano, d1)
            ang_w = abs(wrap180(b0 - b1))
            if ang_w < 0.5 or ang_w > 120:                 # skip degenerate / behind camera
                continue
            g = float(np.linalg.norm(mid - cam) * S)
            fwd = w0 / (2 * np.tan(np.radians(ang_w) / 2))
            r = {"home": home.name, "gt": g, "fw": fwd}
            for s in sources:
                r[s] = sample_depth_sector(depths[s], b_mid, ang_w / 2) if depths[s] is not None else np.nan
            rows.append(r)
    return rows


def main(a):
    root = Path(a.root)
    ids = Path(a.homes).read_text().split() if Path(a.homes).exists() else a.homes.split(",")
    sources = [s.strip() for s in a.sources.split(",")]

    rows = []
    for hid in ids:
        home = root / hid
        for fl in _floors(home):
            rows += _home_rows(home, fl, sources, a.w0)
    if not rows:
        raise SystemExit("no doors with depth — check --sources paths and that depth was generated")

    # PAIRED set: doors where EVERY source has a finite estimate (fair comparison)
    paired = [r for r in rows if all(np.isfinite(r[s]) for s in sources)]
    if not paired:
        raise SystemExit("no doors have depth from ALL sources — did both DAP and PaGeR run on these homes?")
    gt = np.array([r["gt"] for r in paired])

    print("=" * 70)
    print(f"camera->door distance A/B  |  {len(paired)} paired doors over "
          f"{len({r['home'] for r in paired})} homes")
    print("-" * 70)
    print(f"{'source':34}{'median err':>12}{'MAE':>10}")
    res = {}
    for s in sources + ["fw"]:
        est = np.array([r["fw"] if s == "fw" else r[s] for r in paired])
        e = np.abs(est - gt); res[s] = (float(np.median(e)), float(e.mean()))
        name = "fixed-width geometry (depth-free)" if s == "fw" else s
        print(f"{name:34}{np.median(e):10.2f} m{e.mean():8.2f} m")
    print("-" * 70)
    best = min(res, key=lambda s: res[s][0])
    print(f"GT distance median {np.median(gt):.2f} m  |  LOWEST median error: "
          f"{'fixed-width' if best == 'fw' else best}")

    print("\nper-home median error (m):")
    print("  home       " + "".join(f"{s.split('/')[0]:>14}" for s in sources))
    for h in sorted({r["home"] for r in paired}):
        sub = [r for r in paired if r["home"] == h]; g = np.array([r["gt"] for r in sub])
        line = f"  {h:10}"
        for s in sources:
            line += f"{np.median(np.abs(np.array([r[s] for r in sub]) - g)):14.2f}"
        print(line)

    out = config.RESULTS_ROOT / "depth_ab"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "depth_ab.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["home", "gt", "fw"] + sources)
        for r in paired:
            w.writerow([r["home"], r["gt"], r["fw"]] + [r[s] for s in sources])
    fig, ax = plt.subplots(figsize=(6, 6))
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for i, s in enumerate(sources):
        est = np.array([r[s] for r in paired])
        ax.scatter(gt, est, s=10, alpha=.5, c=cols[i % 4],
                   label=f"{s.split('/')[0]} (med {res[s][0]:.2f} m)")
    lim = [0, float(np.percentile(gt, 99)) * 1.1]; ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("GT camera->door distance (m)"); ax.set_ylabel("estimated (m)")
    ax.legend(); ax.set_title(f"depth A/B: camera->door distance ({len(paired)} doors)")
    fig.tight_layout(); fig.savefig(out / "depth_ab.png", dpi=120)
    print(f"\nsaved {out}/depth_ab.csv + depth_ab.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ZInD full_dataset root ($ZIND_ROOT)")
    ap.add_argument("--homes", required=True, help="file of home ids OR comma list")
    ap.add_argument("--sources", default="pager_depth/depth_meters,dap_depth/depth_meters",
                    help="comma list of depth subdirs to compare (first is highlighted first)")
    ap.add_argument("--w0", type=float, default=0.9, help="fixed door-width prior (m) for the baseline")
    main(ap.parse_args())

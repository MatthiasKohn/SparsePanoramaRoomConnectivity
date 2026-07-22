"""
detect_diagnose — WHERE does the door detector lose recall? (decides which recall lever to build)

The connectivity gap (0.913 GT doors -> 0.842 detected) is recall-limited (~0.66). Before building a
fix we diagnose the ~34% MISSED GT doors, over the held-out homes, into actionable buckets:

  near-miss   : a detection exists within [tol, 2*tol] deg -> localization/merge issue
                (undistorted cubemap-face detection is the lever)
  open-missed : truly missed AND the depth shows a GAP at the door bearing (passage recedes vs the
                adjacent wall) -> a GEOMETRIC opening proposal (PaGeR depth) could catch it
  flush-missed: truly missed AND depth is ~flat (closed door flush with the wall) -> no depth signal;
                an appearance ceiling neither lever fixes

Reuses the cached detector output if present (CPU, fast); otherwise runs SegFormer (needs GPU).

  python -m pipelines.detect_diagnose --root $ZIND_ROOT --homes scripts/depth_homes.txt \
      --det_cache results/gtfree/det_cache --depth_sub pager_depth/depth_meters
"""
import argparse, json
from pathlib import Path
import numpy as np

from sparsepano import config
from sparsepano.datasets import zind
from pipelines.connectivity import detect_home, gt_door_az, circ_diff
from pipelines.distance_baseline import sample_depth_sector
from sparsepano.doors import door_dataset


def _floors(home):
    try:
        return list(json.load(open(home / "zind_data.json")).get("merger", {}).keys())
    except Exception:
        return []


def _depth2d(path):
    if not path.exists():
        return None
    d = np.squeeze(np.load(path).astype(np.float32))
    return d if d.ndim == 2 else None


def _open_ratio(depth, b_mid, ang_w):
    """door-sector depth / adjacent-wall depth. >1 => passage recedes (open); ~1 => flush wall."""
    if depth is None:
        return np.nan
    door_d = sample_depth_sector(depth, b_mid, max(ang_w / 2, 3.0))
    off = ang_w / 2 + 12.0
    wall = np.nanmedian([sample_depth_sector(depth, b_mid + off, 4.0),
                         sample_depth_sector(depth, b_mid - off, 4.0)])
    if not (np.isfinite(door_d) and np.isfinite(wall) and wall > 0.1):
        return np.nan
    return door_d / wall


def main(a):
    root = Path(a.root)
    ids = Path(a.homes).read_text().split() if Path(a.homes).exists() else a.homes.split(",")

    factory = None
    if a.det_cache is None or not all(
            (Path(a.det_cache) / f"{h}_det.json").exists() for h in ids):
        import torch
        from sparsepano.doors.door_semantic import SemanticDoorDetector
        dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        factory = lambda: SemanticDoorDetector(device=dev, n_views=a.n_views, fov_deg=a.det_fov)
        print(f"[detector] running SegFormer on {dev} (no full cache found)")
    else:
        print(f"[detector] using cached detections in {a.det_cache}")

    tol = a.tol_deg
    rec = {"total": 0, "matched": 0, "near": 0, "open": 0, "flush": 0, "unknown": 0}
    per_home = {}
    rows = []
    for hid in ids:
        home = root / hid
        for fl_name in _floors(home):
            try:
                fl = zind.ZindFloor(home / "zind_data.json", floor=fl_name)
            except Exception:
                continue
            det_az = detect_home(home, fl, factory, a.det_cache) if factory or a.det_cache else {}
            hm = per_home.setdefault(hid, {"total": 0, "matched": 0})
            for stem in fl.panos:
                gts_mid = [((d0 + d1) / 2, d0, d1) for d0, d1 in fl.panos[stem]["doors_global"]]
                gts_az = gt_door_az(fl, stem)
                dets = det_az.get(stem, [])
                depth = _depth2d(home / a.depth_sub / f"{stem}.npy")
                for k, (g, (mid, d0, d1)) in enumerate(zip(gts_az, gts_mid)):
                    rec["total"] += 1; hm["total"] += 1
                    nearest = min((circ_diff(g, dd) for dd in dets), default=999.0)
                    if nearest < tol:
                        rec["matched"] += 1; hm["matched"] += 1
                        bucket = "matched"
                    elif nearest < 2 * tol:
                        rec["near"] += 1; bucket = "near"
                    else:
                        b0 = door_dataset.door_azimuth(fl, stem, d0)
                        b1 = door_dataset.door_azimuth(fl, stem, d1)
                        ang_w = abs(((b0 - b1 + 180) % 360) - 180)
                        ratio = _open_ratio(depth, g, max(ang_w, 4.0))
                        if not np.isfinite(ratio):
                            rec["unknown"] += 1; bucket = "unknown"
                        elif ratio > a.open_ratio:
                            rec["open"] += 1; bucket = "open"
                        else:
                            rec["flush"] += 1; bucket = "flush"
                    rows.append(dict(home=hid, floor=fl_name, pano=stem, gt_az=round(g, 1),
                                     nearest_det=round(nearest, 1), bucket=bucket))

    T = max(rec["total"], 1)
    print("=" * 66)
    print(f"door-detector recall diagnosis  |  {rec['total']} GT doors, {len(ids)} homes  (tol {tol}°)")
    print("-" * 66)
    print(f"  matched (recall)     : {rec['matched']:4d}  ({rec['matched']/T:.2f})")
    miss = T - rec["matched"]
    print(f"  MISSED               : {miss:4d}  ({miss/T:.2f})")
    print(f"    near-miss (<{2*tol:.0f}°)  : {rec['near']:4d}  -> cubemap-face detection (localization)")
    print(f"    open (depth gap)   : {rec['open']:4d}  -> PaGeR geometric opening proposal CAN catch")
    print(f"    flush (closed)     : {rec['flush']:4d}  -> appearance ceiling; neither lever fixes")
    print(f"    unknown (no depth) : {rec['unknown']:4d}")
    print("-" * 66)
    if miss:
        print(f"lever ceilings on the missed set: near-miss {rec['near']/miss:.0%} | "
              f"open {rec['open']/miss:.0%} | flush {rec['flush']/miss:.0%}")
    out = config.RESULTS_ROOT / "detect_diagnose"; out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out / "misses.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"saved {out}/misses.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--homes", required=True)
    ap.add_argument("--det_cache", default="results/gtfree/det_cache",
                    help="reuse cached detections (CPU) if present; else run SegFormer (GPU)")
    ap.add_argument("--depth_sub", default="pager_depth/depth_meters",
                    help="depth for the open/closed proxy (PaGeR's crisp geometry is best here)")
    ap.add_argument("--tol_deg", type=float, default=15.0)
    ap.add_argument("--open_ratio", type=float, default=1.25,
                    help="door/wall depth ratio above which a missed door is 'open' (a real gap)")
    ap.add_argument("--det_fov", type=float, default=70.0)
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--device", default=None)
    main(ap.parse_args())

"""Write OUR per-pano equirect depth into the exact format SALVe expects, so its RGB-texture BEV path
uses our depth instead of HoHoNet (whose weights are dead). SALVe's loader skips depth inference if the
file already exists, so pre-populating these disables HoHoNet entirely.

Format (from salve/utils/bev_rendering_utils.get_xyzrgb_from_depth):
  depth = imageio.imread(fpath) * 0.001  ->  metres.   So we store depth_mm = metres*1000 as uint16,
  at 512x1024 (H,W) equirect, at {depth_save_root}/{building_id}/{stem}.depth.png.

  python salve_integration/make_salve_depth.py --home $ZIND_ROOT/0021 --building_id 0021 \
      --depth_save_root <dir> --depth_model gt_layout    # or dap / pager (monocular)
"""
import argparse, json, glob
from pathlib import Path
import numpy as np
import imageio

from sparsepano.datasets import zind_floor
from sparsepano.providers import depth as PRO_DEPTH

H, W = 512, 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="ZInD building dir, e.g. $ZIND_ROOT/0021")
    ap.add_argument("--building_id", required=True)
    ap.add_argument("--depth_save_root", required=True)
    ap.add_argument("--depth_model", default="gt_layout", choices=["gt_layout", "fused", "pager", "dap"])
    a = ap.parse_args()
    outdir = Path(a.depth_save_root) / a.building_id
    outdir.mkdir(parents=True, exist_ok=True)
    # discover the building's ACTUAL floors (so no annotated pano is missed -> no HoHoNet fallback)
    jd = json.load(open(Path(a.home) / "zind_data.json"))
    floors = list(jd.get("merger", {}).keys()) or ["floor_01"]

    n = 0
    for floor in floors:
        try:
            fl = zind_floor.ZindFloor(Path(a.home) / "zind_data.json", floor=floor)
        except Exception:
            continue
        for stem in fl.panos:
            fp = outdir / f"{stem}.depth.png"
            if fp.exists():
                continue
            try:
                d = PRO_DEPTH.get_depth(fl, a.home, stem, H, W, model=a.depth_model, mask_doors=False)
            except Exception as e:
                print(f"  [depth] {stem}: {e} -> skip")
                continue
            d = np.asarray(d, np.float32)
            if d.shape != (H, W):
                import cv2
                d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
            imageio.imwrite(fp, mm)
            n += 1
    print(f"[depth] wrote {n} depth maps ({a.depth_model}) -> {outdir}")


if __name__ == "__main__":
    main()

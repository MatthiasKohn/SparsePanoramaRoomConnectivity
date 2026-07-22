"""Map PaGeR inference outputs into this pipeline's per-home depth/normals layout, so the pipeline
reads PaGeR geometry exactly like DAP (via --depth_sub pager_depth/depth_meters).

  python scripts/pager/pager_to_pipeline.py --preds <pager_results> --zind_root $ZIND_ROOT
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="PaGeR --results_path (has depth/preds, normals/preds)")
    ap.add_argument("--zind_root", required=True)
    a = ap.parse_args()
    preds, zr = Path(a.preds), Path(a.zind_root)
    n = 0
    for mod, sub in [("depth", "depth_meters"), ("normals", "normals")]:
        src = preds / mod / "preds"
        if not src.exists():
            print(f"skip {mod}: no {src}"); continue
        for f in sorted(src.glob("*.npz")):
            home, stem = f.stem.split("__", 1)
            arr = np.squeeze(np.load(f)["arr_0"])      # np.savez(path, raw_image) -> key arr_0; squeeze to (H,W)
            out = zr / home / "pager_depth" / sub
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / f"{stem}.npy", arr.astype(np.float32))
            n += 1
    print(f"wrote {n} .npy files under <home>/pager_depth/")


if __name__ == "__main__":
    main()

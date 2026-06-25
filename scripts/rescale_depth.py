import numpy as np, glob, os
from pathlib import Path

RAW_DIR  = Path(r"C:\Users\kohnm\Uni\Promotion\data\Download_immersight_ 2026-06-23_10-23-49\dap_depth\depth_npy")
OUT_DIR  = RAW_DIR.parent / "depth_metric"; OUT_DIR.mkdir(exist_ok=True)
CAM_HEIGHT_M = 1.50          # <-- set to the real tripod/camera height of the capture

for f in glob.glob(str(RAW_DIR / "*.npy")):
    raw = np.load(f).astype(np.float32)          # unscaled DAP output
    H = raw.shape[0]
    nadir = raw[int(0.97*H):, :]                  # bottom 3% of rows = floor straight down
    d_nadir = np.median(nadir[nadir > 0])
    scale = CAM_HEIGHT_M / d_nadir                # per-image metric scale
    np.save(OUT_DIR / os.path.basename(f), raw * scale)
    print(os.path.basename(f), "nadir->", round(float(d_nadir),4), "scale", round(float(scale),1))
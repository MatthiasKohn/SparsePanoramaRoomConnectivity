"""
exp05 — Visualize detected see-through openings on the Immersight chain.

(b) from the discussion: before wiring the doorway into the pose solve, just SHOW
where the geometric detector thinks the openings are, overlaid on the RGB, so the
result can be checked by eye against the real doors. Handles the wrap-around case
(a door split across the left/right image edges, azimuth ~+-180 deg).

Panos (kitchen -> floor -> dining chain):
    30 kitchen, 46 floor, 60 floor (split kitchen door), 37 dining

Run:  python legacy/experiments/exp05_doorway_detect.py
"""
import sys, os
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sparsepano import config
from sparsepano.doors import aperture

IMM = config.DATA_ROOT / "Download_immersight_ 2026-06-23_10-23-49"
MD = IMM / "dap_depth" / "depth_metric"
OUT = config.RESULTS_ROOT / "exp04_immersight"; OUT.mkdir(parents=True, exist_ok=True)

CHAIN = [("30 kitchen", "1273530"), ("46 floor", "1273546"),
         ("60 floor (split door)", "1308360"), ("37 dining", "1273537")]


def rgb(stem, width=1600):
    for ext in (".png", ".jpeg", ".jpg"):
        p = IMM / f"panorama_{stem}{ext}"
        if p.exists():
            im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            return cv2.resize(im, (width, width // 2))
    return None


def depth(stem, h=1024):
    d = np.load(MD / f"panorama_{stem}.npy").astype(np.float32)
    return cv2.resize(d, (h * 2, h), interpolation=cv2.INTER_NEAREST)


def main():
    fig, axes = plt.subplots(len(CHAIN), 1, figsize=(13, 3.0 * len(CHAIN)))
    for ax, (name, stem) in zip(axes, CHAIN):
        im = rgb(stem); Wr = im.shape[1]
        d = depth(stem); Wd = d.shape[1]
        cands = aperture.detect_apertures(d, abs_margin=0.6, rel_margin=0.5)
        ax.imshow(im); ax.set_xlim(0, Wr); ax.set_ylim(im.shape[0], 0)
        f = Wr / Wd
        for c in cands[:6]:
            lo, hi = c.u_lo * f, c.u_hi * f
            spans = [(lo, hi)] if c.u_lo <= c.u_hi else [(lo, Wr), (0, hi)]  # wrap split
            for (x0, x1) in spans:
                ax.add_patch(Rectangle((x0, 0), x1 - x0, im.shape[0], color="#00e5ff",
                                       alpha=0.28, lw=0))
                ax.plot([(x0 + x1) / 2], [im.shape[0] * 0.5], marker="v",
                        color="#d62728", ms=9)
        ax.set_title(f"{name}  —  {len(cands)} see-through openings detected "
                     f"(cyan bands; red = centre direction)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Detected see-through openings on the kitchen→floor→dining chain (Immersight)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "openings_chain.png", dpi=115)
    print("saved", OUT / "openings_chain.png")
    # also print azimuths for the record
    for name, stem in CHAIN:
        d = depth(stem); W = d.shape[1]
        cs = aperture.detect_apertures(d, abs_margin=0.6, rel_margin=0.5)
        az = [round(float((( (c.u_lo+c.u_hi)/2)/W-0.5)*360)) for c in cs[:6]]
        print(f"{name:24} openings az(deg)={az}")


if __name__ == "__main__":
    main()

"""
exp06 — Semantic doors (existence) + depth see-through (geometry), fused & honest.

Labels are NOT open/closed (that needs appearance). Instead:
  green  = door WITH depth see-through geometry (usable pose anchor)
  orange = door, no see-through in depth (pose must come from appearance)
  blue   = window (separate category)
  cyan   = opening with geometry but no semantic door (leaf-less opening / FP)

Real run (laptop GPU):  python experiments/exp06_semantic_doors.py
Self-test (no model):   python experiments/exp06_semantic_doors.py --selftest
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config
from src import aperture
from src.doors import Door, fuse
from src.door_semantic import SemanticDoorDetector

IMM = config.DATA_ROOT / "Download_immersight_ 2026-06-23_10-23-49"
MD = IMM / "dap_depth" / "depth_metric"
OUT = config.RESULTS_ROOT / "exp04_immersight"; OUT.mkdir(parents=True, exist_ok=True)
CHAIN = [("30 kitchen", "1273530"), ("46 floor", "1273546"),
         ("60 floor", "1308360"), ("37 dining", "1273537")]


def style(d):
    if d.category == "window":  return "#1f77b4", "window"
    if d.category == "opening": return "#00bcd4", "opening?"
    if d.seethrough:            return "#2ca02c", "door+geom"
    return "#ff7f0e", "door"


def rgb(stem, width=1500):
    for ext in (".png", ".jpeg", ".jpg"):
        p = IMM / f"panorama_{stem}{ext}"
        if p.exists():
            return cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (width, width // 2))
    return None


def depth_apertures(stem, pano_id):
    p = MD / f"panorama_{stem}.npy"
    if not p.exists():
        return []
    d = cv2.resize(np.load(p).astype(np.float32), (2048, 1024), interpolation=cv2.INTER_NEAREST)
    out = []
    for c in aperture.detect_apertures(d, abs_margin=0.6, rel_margin=0.4):
        ext = ((c.u_hi - c.u_lo) % 2048) / 2048 * 360
        out.append(Door(pano_id, float(np.degrees(c.center_az)), float(ext), float(c.score),
                        source="depth", seethrough=True))
    return out


def make_detector(stem, selftest):
    if selftest:
        def fake(img):
            H, W = img.shape[:2]; lab = np.zeros((H, W), int)
            lab[int(.35*H):int(.95*H), int(.46*W):int(.54*W)] = 14   # door to floor
            lab[int(.15*H):int(.32*H), int(.70*W):int(.80*W)] = 8    # window high
            return lab
        return SemanticDoorDetector(stem, n_views=6, segmenter=fake)
    return SemanticDoorDetector(stem, n_views=8)


def band(ax, az, ext, color, W, H, label):
    x = (az / 360 + 0.5) * W
    if not (0 <= x <= W):
        return
    w = max(ext / 360 * W, 7)
    ax.add_patch(Rectangle((x - w/2, 0), w, H, color=color, alpha=.30, lw=0))
    ax.plot([x], [H*0.5], marker="v", color=color, ms=9)
    ax.text(x, H*0.07, label, color="white", fontsize=6.5, ha="center",
            bbox=dict(boxstyle="round,pad=0.1", fc=color, ec="none", alpha=.85))


def main(selftest=False):
    fig, axes = plt.subplots(len(CHAIN), 1, figsize=(13, 3.0 * len(CHAIN)))
    for ax, (name, stem) in zip(axes, CHAIN):
        im = rgb(stem); H, W = im.shape[:2]
        sem = make_detector(stem, selftest).detect(im)
        doors = fuse(sem, depth_apertures(stem, stem))
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for d in doors:
            col, lab = style(d)
            band(ax, d.azimuth_deg, max(d.az_extent_deg, 8), col, W, H, lab)
        nd = sum(d.category == "door" for d in doors)
        ng = sum(d.category == "door" and d.seethrough for d in doors)
        nw = sum(d.category == "window" for d in doors)
        ax.set_title(f"{name}: {nd} doors ({ng} with depth-geometry), {nw} windows   "
                     f"green=door+geom · orange=door · blue=window · cyan=opening", fontsize=10)
    fig.suptitle("Doors (semantic) + see-through (depth), no open/closed claim (Immersight)"
                 + ("  [SELF-TEST]" if selftest else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    nm = "doors_fused_selftest.png" if selftest else "doors_fused.png"
    fig.savefig(OUT / nm, dpi=115); print("saved", OUT / nm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    main(ap.parse_args().selftest)

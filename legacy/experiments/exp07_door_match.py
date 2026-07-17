"""
exp07 — Match doors across two panoramas = first connectivity edge.

Detect doors in pano A and pano B (semantic), embed each door crop (DINOv2),
keep mutual-nearest matches above threshold, and draw the matched doorways.
A match between two panos IS a connectivity edge (and the future pose anchor).

Real run (laptop GPU):  python legacy/experiments/exp07_door_match.py            # kitchen<->floor
Self-test (no models):  python legacy/experiments/exp07_door_match.py --selftest
"""
import sys, os, argparse
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.doors.doors import Door
from sparsepano.doors.door_semantic import SemanticDoorDetector
from sparsepano.pose.matching import DoorMatcher, door_crop

IMM = config.DATA_ROOT / "Download_immersight_ 2026-06-23_10-23-49"
OUT = config.RESULTS_ROOT / "exp04_immersight"; OUT.mkdir(parents=True, exist_ok=True)
PAIR = [("30 kitchen", "1273530"), ("46 floor", "1273546")]
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def rgb(stem, width=1500):
    for ext in (".png", ".jpeg", ".jpg"):
        p = IMM / f"panorama_{stem}{ext}"
        if p.exists():
            return cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (width, width // 2))


def detectors_and_matcher(selftest):
    if selftest:
        def fake_seg(img):
            H, W = img.shape[:2]; lab = np.zeros((H, W), int)
            lab[int(.4*H):int(.95*H), int(.46*W):int(.54*W)] = 14
            return lab
        det = lambda stem: SemanticDoorDetector(stem, n_views=6, segmenter=fake_seg)
        match = DoorMatcher(embed=lambda img: np.array([1.0, 0.0]), min_sim=0.5)
    else:
        det = lambda stem: SemanticDoorDetector(stem, n_views=8)
        match = DoorMatcher()
    return det, match


def main(selftest=False):
    det, matcher = detectors_and_matcher(selftest)
    (nA, sA), (nB, sB) = PAIR
    imA, imB = rgb(sA), rgb(sB)
    dA = [d for d in det(sA).detect(imA) if d.category == "door"]
    dB = [d for d in det(sB).detect(imB) if d.category == "door"]
    matches = matcher.match(imA, dA, imB, dB)
    print(f"{nA}: {len(dA)} doors | {nB}: {len(dB)} doors | matches: {len(matches)}")
    for k, m in enumerate(matches):
        print(f"  edge {k}: A az {m.az_a:.0f} <-> B az {m.az_b:.0f}  sim {m.score:.2f}")

    fig, ax = plt.subplots(2, 1, figsize=(13, 6))
    for axi, im, name, doors in [(ax[0], imA, nA, dA), (ax[1], imB, nB, dB)]:
        axi.imshow(im); axi.set_xticks([]); axi.set_yticks([]); axi.set_title(name, fontsize=11)
        W = im.shape[1]
        for d in doors:
            x = (d.azimuth_deg / 360 + 0.5) * W
            axi.axvline(x, color="gray", lw=1, alpha=.5)
    for k, m in enumerate(matches):
        c = COLORS[k % 10]
        for axi, im, d in [(ax[0], imA, dA[m.a_idx]), (ax[1], imB, dB[m.b_idx])]:
            x = (d.azimuth_deg / 360 + 0.5) * im.shape[1]
            axi.axvline(x, color=c, lw=3)
            axi.text(x, 20, f"#{k}", color="white", fontsize=10, ha="center",
                     bbox=dict(boxstyle="round", fc=c, ec="none"))
    fig.suptitle(f"Door matching = connectivity edge ({nA} ↔ {nB})"
                 + ("  [SELF-TEST]" if selftest else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    nm = "door_match_selftest.png" if selftest else "door_match.png"
    fig.savefig(OUT / nm, dpi=120); print("saved", OUT / nm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    main(ap.parse_args().selftest)

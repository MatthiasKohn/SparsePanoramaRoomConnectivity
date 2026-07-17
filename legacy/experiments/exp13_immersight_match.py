"""
exp13 — Try the TRAINED door encoder on Immersight (out-of-distribution sanity check).

Detect doors in each chain pano (SegFormer), crop each (fov 70, as in training),
embed with the trained encoder, and match doors ACROSS panos. Output is built for
eyeballing: every high-similarity cross-pano door match is shown as a side-by-side
crop pair with its cosine -- you decide if it's really the same doorway.

  python legacy/experiments/exp13_immersight_match.py --ckpt runs/full/best.pt
  python legacy/experiments/exp13_immersight_match.py --ckpt door_encoder.pt --stems 1273530,1273546,1308360,1273537
  python legacy/experiments/exp13_immersight_match.py --selftest        # plumbing, no models
"""
import sys, os, argparse
from pathlib import Path
from itertools import combinations
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.geometry import panoproj
from sparsepano.doors import contrastive
from sparsepano.doors.door_semantic import SemanticDoorDetector

IMM = config.DATA_ROOT / "Download_immersight_ 2026-06-23_10-23-49"
OUT = config.RESULTS_ROOT / "exp04_immersight"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = {"1273530": "kitchen", "1273546": "floor", "1308360": "floor2", "1273537": "dining"}


def load_rgb(stem):
    for ext in (".png", ".jpeg", ".jpg"):
        p = IMM / f"panorama_{stem}{ext}"
        if p.exists():
            return cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (4096, 2048)), cv2.COLOR_BGR2RGB)
    return None


def make_detector(stem, selftest):
    if selftest:
        def fake(img):
            H, W = img.shape[:2]; lab = np.zeros((H, W), int)
            lab[int(.4*H):int(.95*H), int(.45*W):int(.55*W)] = 14
            return lab
        return SemanticDoorDetector(stem, n_views=4, segmenter=fake)
    return SemanticDoorDetector(stem, n_views=8)


def main(stems, ckpt, selftest=False, fov=70, min_sim=0.0):
    embed = ((lambda c: np.random.default_rng(abs(hash(c.tobytes())) % 2**32).standard_normal(128))
             if selftest else contrastive.load_embedder(ckpt, "cuda" if _cuda() else "cpu"))
    # detect + crop + embed doors per pano
    doors = {}   # stem -> list of (azimuth, crop, emb)
    for s in stems:
        im = load_rgb(s)
        if im is None:
            print(f"  [skip] pano {s} not found"); continue
        dets = [d for d in make_detector(s, selftest).detect(im) if d.category == "door"]
        lst = []
        for d in dets:
            crop = panoproj.e2p(im, d.azimuth_deg, 0, fov, (224, 224))
            e = np.asarray(embed(crop), float); e /= np.linalg.norm(e) + 1e-8
            lst.append((d.azimuth_deg, crop, e))
        doors[s] = lst
        print(f"  {s} ({NAMES.get(s,'?')}): {len(lst)} doors")

    # cross-pano best match per pano-pair + all candidate matches
    matches = []          # (sim, si, di, sj, dj)
    print(f"\n{'pano-pair':22} {'best cos':>8}")
    for si, sj in combinations([s for s in stems if doors.get(s)], 2):
        best = (-1, None, None)
        for di, (_, _, ei) in enumerate(doors[si]):
            for dj, (_, _, ej) in enumerate(doors[sj]):
                sim = float(ei @ ej)
                if sim >= min_sim:
                    matches.append((sim, si, di, sj, dj))
                if sim > best[0]:
                    best = (sim, di, dj)
        print(f"{NAMES.get(si,si)+'<->'+NAMES.get(sj,sj):22} {best[0]:8.2f}")

    matches.sort(reverse=True)
    top = matches[:8]
    if top:
        fig, axes = plt.subplots(len(top), 2, figsize=(5, 2.4 * len(top)))
        if len(top) == 1:
            axes = axes[None, :]
        for row, (sim, si, di, sj, dj) in enumerate(top):
            for col, (s, d) in enumerate([(si, di), (sj, dj)]):
                ax = axes[row, col]; ax.imshow(doors[s][d][1]); ax.axis("off")
                ax.set_title(f"{NAMES.get(s,s)} @{doors[s][d][0]:.0f}deg"
                             + (f"\ncos={sim:.2f}" if col == 1 else ""), fontsize=8)
        fig.suptitle("Immersight cross-pano door matches (trained encoder) — same door?", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = OUT / ("imm_match_selftest.png" if selftest else "imm_match.png")
        fig.savefig(out, dpi=120); print("\nsaved", out)
    else:
        print("no matches above threshold")


def _cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="door_encoder.pt")
    ap.add_argument("--stems", default="1273530,1273546,1308360,1273537")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--min_sim", type=float, default=0.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    main(a.stems.split(","), a.ckpt, a.selftest, a.fov, a.min_sim)

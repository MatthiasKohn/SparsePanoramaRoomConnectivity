"""
exp15 — What does the encoder LOOK AT when it matches two cross-view door crops?

Occlusion sensitivity: slide a gray patch over a crop, measure how much
cos(f(crop_occluded), f(partner)) DROPS. Big drop => that region drives the match.

For ZInD pairs (--data) we also PAINT the saliency back onto the FULL panoramas, so you
can see where the crop sits and whether it is really on a matching doorway.

  python experiments/exp15_match_saliency.py --data data_doorpairs --idx 7 --ckpt door_encoder.pt
  python experiments/exp15_match_saliency.py --pair cropA.png cropB.png --ckpt door_encoder.pt
  python experiments/exp15_match_saliency.py --data data_doorpairs --idx 3 --selftest
"""
import sys, os, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src import contrastive, panoproj

FOV = 70.0


def occ_saliency(img, partner_vec, embed, grid=12, occ_frac=0.22):
    H, W = img.shape[:2]
    base = float(embed(img) @ partner_vec)
    k = max(8, int(occ_frac * H)); sal = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            cy, cx = int((i + .5) / grid * H), int((j + .5) / grid * W)
            im2 = img.copy()
            im2[max(0, cy-k//2):cy+k//2, max(0, cx-k//2):cx+k//2] = 128
            sal[i, j] = base - float(embed(im2) @ partner_vec)
    sal = cv2.resize(np.clip(sal, 0, None), (W, H))
    return sal / (sal.max() + 1e-8), base


def paint_on_pano(pano, sal_crop, az, fov=FOV):
    """Scatter the crop saliency onto the equirect pano via the e2p sampling map."""
    H, W = pano.shape[:2]; Hc, Wc = sal_crop.shape
    u, v = panoproj.e2p_uvmap(H, W, az, 0, fov, (Hc, Wc))
    ui = np.clip(u.round().astype(int), 0, W - 1); vi = np.clip(v.round().astype(int), 0, H - 1)
    heat = np.zeros((H, W)); cnt = np.zeros((H, W))
    np.add.at(heat, (vi, ui), sal_crop); np.add.at(cnt, (vi, ui), 1)
    heat = heat / np.maximum(cnt, 1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=W * 0.004)
    return heat / (heat.max() + 1e-8)


def get_pair(a):
    if a.pair:
        A = cv2.cvtColor(cv2.imread(a.pair[0]), cv2.COLOR_BGR2RGB)
        B = cv2.cvtColor(cv2.imread(a.pair[1]), cv2.COLOR_BGR2RGB)
        return dict(A=A, B=B, tag=Path(a.pair[0]).stem)
    r = list(csv.DictReader(open(Path(a.data) / "pairs.csv")))[a.idx]
    crops = Path(a.data) / "crops"
    A = cv2.cvtColor(cv2.imread(str(crops / r["crop_a"])), cv2.COLOR_BGR2RGB)
    B = cv2.cvtColor(cv2.imread(str(crops / r["crop_b"])), cv2.COLOR_BGR2RGB)
    out = dict(A=A, B=B, tag=r["door_id"])
    pdir = config.zind_paths()["panos"]
    pa, pb = pdir / f"{r['pano_a']}.jpg", pdir / f"{r['pano_b']}.jpg"
    if pa.exists() and pb.exists():
        out["panoA"] = cv2.cvtColor(cv2.resize(cv2.imread(str(pa)), (2048, 1024)), cv2.COLOR_BGR2RGB)
        out["panoB"] = cv2.cvtColor(cv2.resize(cv2.imread(str(pb)), (2048, 1024)), cv2.COLOR_BGR2RGB)
        out["az_a"], out["az_b"] = float(r["az_a"]), float(r["az_b"])
    return out


def main(a):
    P = get_pair(a)
    A = cv2.resize(P["A"], (224, 224)); B = cv2.resize(P["B"], (224, 224))
    if a.selftest:
        embed = lambda im: (lambda v: v/(np.linalg.norm(v)+1e-8))(
            np.random.default_rng(abs(hash(im.tobytes())) % 2**32).standard_normal(128))
    else:
        import torch
        embed = contrastive.load_embedder(a.ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    fA = embed(A); fA /= np.linalg.norm(fA)+1e-8
    fB = embed(B); fB /= np.linalg.norm(fB)+1e-8
    salA, base = occ_saliency(A, fB, embed, a.grid)
    salB, _ = occ_saliency(B, fA, embed, a.grid)
    print(f"door {P['tag']}: cos(A,B) = {base:.3f}")
    out = config.RESULTS_ROOT / "saliency"; out.mkdir(parents=True, exist_ok=True)
    ckpt_tag = "selftest" if a.selftest else Path(a.ckpt).stem

    if "panoA" in P:                                   # full-pano context view
        hA = paint_on_pano(P["panoA"], salA, P["az_a"])
        hB = paint_on_pano(P["panoB"], salB, P["az_b"])
        fig = plt.figure(figsize=(13, 8)); gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 2])
        for row, (pano, h, az, lbl) in enumerate([(P["panoA"], hA, P["az_a"], "pano A"),
                                                  (P["panoB"], hB, P["az_b"], "pano B")]):
            ax = fig.add_subplot(gs[row, :]); ax.imshow(pano)
            ax.imshow(h, cmap="jet", alpha=0.45); ax.axvline((az/360+0.5)*pano.shape[1], color="w", lw=1)
            ax.set_title(f"{lbl} (door az {az:.0f}deg) — saliency on the FULL pano", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        for col, (crop, sal, lbl) in enumerate([(A, salA, "crop A"), (B, salB, "crop B")]):
            ax = fig.add_subplot(gs[2, col]); ax.imshow(crop); ax.imshow(sal, cmap="jet", alpha=0.45)
            ax.set_title(lbl, fontsize=9); ax.axis("off")
        fig.suptitle(f"Cross-view match saliency  (cos={base:.2f})"
                     + ("  [SELF-TEST]" if a.selftest else ""), fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = out / (f"saliency_pano_{P['tag']}_{ckpt_tag}.png")
    else:                                              # crop-only (e.g. --pair)
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
        ax[0].imshow(A); ax[0].set_title(f"A (cos={base:.2f})", fontsize=9); ax[0].axis("off")
        ax[1].imshow(A); ax[1].imshow(salA, cmap="jet", alpha=.45); ax[1].set_title("A saliency"); ax[1].axis("off")
        ax[2].imshow(B); ax[2].imshow(salB, cmap="jet", alpha=.45); ax[2].set_title("B saliency"); ax[2].axis("off")
        ax[3].imshow(B); ax[3].set_title("B", fontsize=9); ax[3].axis("off")
        p = out / (f"saliency_{P['tag']}_{ckpt_tag}.png")
    fig.savefig(p, dpi=120); print("saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_doorpairs")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--pair", nargs=2)
    ap.add_argument("--ckpt", default="door_encoder.pt")
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    main(a)

"""
Occlusion saliency for the cross-view door embedding — visualized ON THE PANORAMA.

Method (as on the slide, done right): slide a small gray patch over the door crop of pano A;
at each position re-embed the occluded crop and measure how much its similarity to the
MATCHING crop of pano B drops. Large drop = that region makes the door recognisable across
rooms. The crop is a 70°-FOV perspective view (e2p) centred on the door; we map the saliency
back through `e2p_uvmap` onto the equirectangular pano, so the figure shows the whole room
with a heatmap glowing at the doorway (+ zoom insets of the matched pair) — not a lonely crop.

Runs where the model runs (torch + DINOv2 backbone). On Leonardo `source scripts/env_leonardo.sh`
first (sets DINOV2_REPO so no GitHub fetch, ZIND_ROOT for the panos, PYTHONPATH).

  python -m pipelines.door_saliency --n 6 --patch 24 --stride 12
  # optional: --context_deg 120  (crop the pano to +/-120 deg around the door -> door not tiny)
"""
import os, csv, argparse
from pathlib import Path
import numpy as np
import cv2

from sparsepano.geometry import panoproj

PW, PH = 4096, 2048                                   # pano working size (matches crop extraction)
IMEAN = np.array([0.485, 0.456, 0.406], np.float32)
ISTD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- model / embedding
def _load_encoder(weights, device):
    import torch
    from sparsepano.doors.contrastive import build_encoder
    enc = build_encoder(device=device)
    ckpt = torch.load(weights, map_location=device)
    for k in ("model", "state_dict", "encoder"):
        if isinstance(ckpt, dict) and k in ckpt:
            ckpt = ckpt[k]; break
    missing, unexpected = enc.load_state_dict(ckpt, strict=False)
    print(f"[saliency] loaded {weights} (missing={len(missing)} unexpected={len(unexpected)})")
    enc.eval()
    return enc


def _prep(img_bgr, size=224):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((rgb - IMEAN) / ISTD).transpose(2, 0, 1)


def _embed(enc, arrs, device, bs=64):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(arrs), bs):
            x = torch.tensor(np.stack(arrs[i:i + bs]), device=device)
            out.append(enc(x).cpu().numpy())
    return np.concatenate(out, 0)


def saliency_for_crop(enc, a_bgr, b_bgr, device, size=224, patch=24, stride=12, gray=0.5):
    """Saliency of crop A w.r.t. its match B (cosine-similarity drop under occlusion)."""
    base = _prep(a_bgr, size)
    zb = _embed(enc, [_prep(b_bgr, size)], device)[0]
    base_sim = float(_embed(enc, [base], device)[0] @ zb)
    grayv = ((gray - IMEAN) / ISTD)[:, None, None]
    pos, occ = [], []
    for y in range(0, size - 1, stride):
        for x in range(0, size - 1, stride):
            v = base.copy(); v[:, y:y + patch, x:x + patch] = grayv
            occ.append(v); pos.append((y, x))
    sims = _embed(enc, occ, device) @ zb
    sal = np.zeros((size, size), np.float32); cnt = np.zeros((size, size), np.float32)
    for (y, x), s in zip(pos, sims):
        sal[y:y + patch, x:x + patch] += (base_sim - float(s)); cnt[y:y + patch, x:x + patch] += 1
    sal = np.clip(sal / np.maximum(cnt, 1e-6), 0, None)
    return (sal / max(sal.max(), 1e-9)), base_sim


# ---------------------------------------------------------------- pano overlay
def _heat(sal01):
    return cv2.applyColorMap((np.clip(sal01, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)


def paint_on_pano(pano_bgr, sal, az, fov, size=224):
    """Map the 224² crop-saliency back onto the equirect pano via e2p_uvmap, return overlay."""
    pano = cv2.resize(pano_bgr, (PW, PH))
    u, v = panoproj.e2p_uvmap(PH, PW, az, 0.0, fov, (size, size))
    ui = np.clip(np.round(u).astype(int), 0, PW - 1)
    vi = np.clip(np.round(v).astype(int), 0, PH - 1)
    acc = np.zeros((PH, PW), np.float32); cov = np.zeros((PH, PW), np.float32)
    np.add.at(acc, (vi, ui), sal); np.add.at(cov, (vi, ui), 1.0)
    # fill the gaps between the 224² samples with a blur, normalized by blurred coverage
    acc = cv2.GaussianBlur(acc, (0, 0), 7); covb = cv2.GaussianBlur(cov, (0, 0), 7)
    sal_eq = np.where(covb > 0.02, acc / np.maximum(covb, 1e-6), 0.0)
    sal_eq = sal_eq / max(sal_eq.max(), 1e-9)
    a = np.clip(sal_eq * 1.15, 0, 1)[..., None]
    over = (pano * (1 - 0.6 * a) + _heat(sal_eq) * (0.6 * a)).astype(np.uint8)
    # doorway footprint outline (crop border mapped to pano)
    border = np.concatenate([np.stack([u[0], v[0]], 1), np.stack([u[:, -1], v[:, -1]], 1),
                             np.stack([u[-1][::-1], v[-1][::-1]], 1),
                             np.stack([u[:, 0][::-1], v[:, 0][::-1]], 1)]).astype(np.int32)
    if border[:, 0].max() - border[:, 0].min() < PW * 0.5:      # skip if it wraps the seam
        cv2.polylines(over, [border], True, (255, 255, 255), 3)
    return over, (int(u.mean()), int(v.mean()))


def _context_crop(over, center_u, context_deg):
    if not context_deg:
        return over
    half = int(context_deg / 360.0 * PW / 2)
    x0 = center_u - half
    idx = (np.arange(x0, x0 + 2 * half) % PW)
    return over[:, idx]


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main(a):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = _load_encoder(a.weights, device)
    crops, out = Path(a.crops), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    zind = Path(a.zind_root)
    rows = list(csv.DictReader(open(a.pairs)))[: a.n]

    for r in rows:
        a_bgr = cv2.imread(str(crops / r["crop_a"])); b_bgr = cv2.imread(str(crops / r["crop_b"]))
        home = a.home or r["scene"].split("_")[0]
        pano_a, tried = None, []
        for hh in (home, home.zfill(4), r["scene"], r["scene"].zfill(4)):
            cand = zind / hh / "panos" / f"{r['pano_a']}.jpg"
            tried.append(str(cand))
            if cand.exists():
                pano_a = cv2.imread(str(cand)); break
        if a_bgr is None or b_bgr is None or pano_a is None:
            print(f"skip {r['door_id']}: pano not found (set --home / --zind_root). tried e.g. {tried[0]}")
            continue

        sal, sim = saliency_for_crop(enc, a_bgr, b_bgr, device, a.size, a.patch, a.stride)
        over, ctr = paint_on_pano(pano_a, sal, float(r["az_a"]), a.fov, a.size)
        over = _context_crop(over, ctr[0], a.context_deg)

        # scale pano figure to a presentable width; add matched-pair insets underneath
        Wt = a.out_w; Ht = int(over.shape[0] * Wt / over.shape[1])
        fig = cv2.resize(over, (Wt, Ht))
        z = int(Wt * 0.24)
        iA = _label(cv2.resize(cv2.addWeighted(cv2.resize(a_bgr, (a.size, a.size)), 0.55,
                    _heat(sal), 0.45, 0), (z, z)), "room A: door saliency")
        iB = _label(cv2.resize(cv2.resize(b_bgr, (a.size, a.size)), (z, z)), "room B: same door")
        strip = np.full((z + 8, Wt, 3), 255, np.uint8)
        strip[4:4 + z, 8:8 + z] = iA; strip[4:4 + z, 16 + z:16 + 2 * z] = iB
        cv2.putText(strip, f"{r['door_id']}   match similarity = {sim:.2f}",
                    (24 + 2 * z, z // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        panel = np.concatenate([fig, strip], axis=0)
        p = out / f"pano_saliency_{r['door_id']}.png"; cv2.imwrite(str(p), panel)
        print("wrote", p, f"(sim {sim:.2f})")

    sheets = sorted(out.glob("pano_saliency_*.png"))[: a.n]
    if sheets:
        tiles = [cv2.imread(str(p)) for p in sheets]
        w = min(t.shape[1] for t in tiles)
        tiles = [cv2.resize(t, (w, int(t.shape[0] * w / t.shape[1]))) for t in tiles]
        cv2.imwrite(str(out / "contact_sheet.png"), np.concatenate(tiles, 0))
        print("wrote", out / "contact_sheet.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="results/door_pairs_demo/crops")
    ap.add_argument("--pairs", default="results/door_pairs_demo/pairs.csv")
    ap.add_argument("--weights", default="weights/best_hardneg.pt")
    ap.add_argument("--zind_root", default=os.environ.get("ZIND_ROOT", "../data/zind/full_dataset"))
    ap.add_argument("--home", default="", help="force the ZInD home folder for the panos (e.g. 0000)")
    ap.add_argument("--out", default="results/door_pairs_demo/saliency")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--patch", type=int, default=24, help="occluder size (px @ 224); smaller = finer")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--fov", type=float, default=70.0, help="MUST match crop extraction fov")
    ap.add_argument("--context_deg", type=float, default=0.0,
                    help="crop pano to +/- this many deg around the door (0 = full 360)")
    ap.add_argument("--out_w", type=int, default=1600, help="output figure width (px)")
    a = ap.parse_args()
    main(a)

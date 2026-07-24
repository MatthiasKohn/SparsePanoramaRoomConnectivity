"""
Oracle floor — the GT UPPER-BOUND run (Paper-2, stage 2a: assembly, no generation yet).

Uses every ground truth we have to build the best possible navigable floor, so later we can
substitute one component at a time (GT pose -> SALVe/BADGR, GT layout-depth -> PaGeR, GT doors
-> detector, +generation) and measure how far each drop falls from THIS ceiling.

All GT here:
  - poses:      ZInD floor_plan_transformation (GT camera poses)
  - geometry:   per-pano depth rendered from the GT room LAYOUT (sparsepano.geometry.layout_depth)
  - one real pano per room as the appearance source
Assembly: per-room 3D Gaussians (pano + layout-depth) placed at GT poses -> one floor.

Evaluation (metric set frozen for ALL later substitutions):
  (A) held-out novel-view quality — for rooms with >=2 panos, rebuild the floor WITHOUT one
      pano, render perspective tiles at its GT pose, score vs the real pano's e2p tiles:
      PSNR / SSIM / LPIPS on covered pixels.
  (B) geometric — coverage / disocclusion (alpha holes) of those held-out views.
  (C) qualitative — a through-floor walkthrough (perspective frames) + merged .ply.
Layout error (Umeyama camera RMSE) is 0 here by construction; the harness (`pose_rmse`) is
wired for the substitution runs.

Needs torch + gsplat (GPU). On Leonardo: `source scripts/env_leonardo.sh` (GT panos in ZIND_ROOT,
DINOV2 not needed here). Run:
  python -m pipelines.oracle_floor --home $ZIND_ROOT/0072 --floor floor_02 --tag oracle_0072_f2
"""
import os, csv, argparse
from pathlib import Path
import numpy as np
import cv2

from sparsepano import config
from sparsepano.datasets import zind_floor, zind
from sparsepano.geometry import layout_depth, panoproj
from sparsepano.gs import gs_optim
from pipelines.gs_room_prototype import build_room_gaussians, merge


# --------------------------------------------------------------------- data
def load_floor(home, floor):
    fl = zind_floor.ZindFloor(Path(home) / "zind_data.json", floor=floor)
    meters = float(fl.meters_per_coord)
    panos = [p for p in fl.panos if len(np.asarray(fl.panos[p]["verts_global"])) >= 3]
    return fl, meters, panos


def _pano_rgb(home, stem, hw):
    p = Path(home) / "panos" / f"{stem}.jpg"
    im = cv2.imread(str(p))
    if im is None:
        return None
    return cv2.cvtColor(cv2.resize(im, (hw[1], hw[0])), cv2.COLOR_BGR2RGB)


def _cull_depth_edges(depth, rel=0.15):
    """Zero out depth at large discontinuities (wall<->ceiling/floor seams, opening edges) so we
    don't splat 'rings' of points across those jumps. Zeroed pixels are skipped by backprojection."""
    d = depth.copy()
    gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
    gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
    thr = np.maximum(0.25, rel * d)                       # >=0.25 m or 15% of range
    d[(gx > thr) | (gy > thr)] = 0.0
    return d


def room_gaussians(fl, meters, home, stem, H, W, stride, max_depth, scale_mult, cull=True):
    depth = layout_depth.render_layout_depth(fl, stem, H, W, max_depth=max_depth)
    if cull:
        depth = _cull_depth_edges(depth)
    rgb = _pano_rgb(home, stem, (H, W))
    if rgb is None:
        return None, None
    pose = zind._pose_c2w(fl.panos[stem], meters)
    g = build_room_gaussians(rgb, depth, pose, stride=stride, max_depth=max_depth, scale_mult=scale_mult)
    return g, pose


def build_floor(fl, meters, home, stems, H, W, stride, max_depth, scale_mult):
    gs, rgbs, poses = [], [], []
    for s in stems:
        g, pose = room_gaussians(fl, meters, home, s, H, W, stride, max_depth, scale_mult)
        if g is None:
            continue
        gs.append(g); poses.append(pose)
        rgbs.append(_pano_rgb(home, s, (H, W)))
    return (merge(gs) if gs else None), rgbs, poses


# --------------------------------------------------------------------- metrics
def _psnr(a, b, m):
    a, b = a[m].astype(np.float32), b[m].astype(np.float32)
    if a.size == 0:
        return float("nan")
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-6 else float(10 * np.log10(255.0 ** 2 / mse))


def _ssim(a, b):
    """Grayscale Gaussian-window SSIM in numpy (no skimage dependency)."""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11); s = 1.5
    mu_a = cv2.GaussianBlur(ga, k, s); mu_b = cv2.GaussianBlur(gb, k, s)
    va = cv2.GaussianBlur(ga * ga, k, s) - mu_a ** 2
    vb = cv2.GaussianBlur(gb * gb, k, s) - mu_b ** 2
    vab = cv2.GaussianBlur(ga * gb, k, s) - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / \
           ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2) + 1e-12)
    return float(np.clip(ssim, -1, 1).mean())


class _LPIPS:
    def __init__(self, device):
        self.fn = None
        try:
            import lpips, torch
            self.fn = lpips.LPIPS(net="alex").to(device).eval(); self.torch = torch; self.device = device
        except Exception as e:
            print(f"[oracle] LPIPS unavailable ({e}); skipping LPIPS")

    def __call__(self, a, b):
        if self.fn is None:
            return float("nan")
        t = self.torch
        def to(x): return t.tensor(x.transpose(2, 0, 1)[None] / 127.5 - 1, dtype=t.float32, device=self.device)
        with t.no_grad():
            return float(self.fn(to(a), to(b)).item())


def pose_rmse(gt_c2w, est_c2w):
    """Umeyama-aligned camera-centre RMSE (m). 0 in the oracle; used under substitution."""
    G = np.array([p[:3, 3] for p in gt_c2w]); E = np.array([p[:3, 3] for p in est_c2w])
    if len(G) < 3:
        return float("nan")
    Gc, Ec = G - G.mean(0), E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec.T @ Gc)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = U @ Vt
    s = S.sum() / (Ec ** 2).sum()
    aligned = s * (Ec @ R) + G.mean(0)
    return float(np.sqrt(np.mean(np.sum((aligned - G) ** 2, 1))))


# --------------------------------------------------------------------- rendering
def render_tiles(g, pose_c2w, basis, vflip, yaws, fov, size, device):
    """Perspective tiles at a pose (yaw sweep, pitch 0), returned UPRIGHT. gsplat's render is
    vertically mirrored when the convention has vflip=True, so we flip it back here."""
    from pipelines.gs_room_prototype import gsplat_render
    out = []
    for y in yaws:
        c2w = pose_c2w.copy()
        c2w[:3, :3] = pose_c2w[:3, :3] @ gs_optim._Ry(np.radians(y))
        rgb, alpha = gsplat_render(g, c2w, basis, fov, size, device)
        if vflip:
            rgb, alpha = rgb[::-1].copy(), alpha[::-1].copy()
        out.append((rgb, alpha))
    return out


def _label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------- main
def main(a):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fl, meters, panos = load_floor(a.home, a.floor)
    print(f"[oracle] {Path(a.home).name}/{a.floor}: {len(panos)} panos with layout")
    out = config.RESULTS_ROOT / "oracle_floor" / a.tag; out.mkdir(parents=True, exist_ok=True)
    H, W = a.gs_h, a.gs_h * 2
    yaws = list(np.linspace(0, 360, a.yaws, endpoint=False))
    lp = _LPIPS(device)

    # ONE real pano per room (target setting) = the floor input; the rest are held-out.
    by_room = {}
    for s in panos:
        by_room.setdefault(fl.panos[s]["room"], []).append(s)
    inputs = [ss[0] for ss in by_room.values()]                # one primary pano per room
    extras = [s for ss in by_room.values() for s in ss[1:]]    # held-out novel views
    print(f"[oracle] {len(inputs)} rooms -> floor from 1 pano/room; {len(extras)} held-out views")

    full, rgbs, poses = build_floor(fl, meters, a.home, inputs, H, W, a.stride, a.max_depth, a.scale_mult)
    if full is None:
        raise SystemExit("no panos could be loaded — check --home / that GT panos exist there")
    from sparsepano.gs import gsplat_init as gi
    gi.write_point_ply(str(out / "floor.ply"), full)

    # convention from ONE room only (avoids multi-room bleed -> a clean, high-PSNR basis)
    g0, p0 = room_gaussians(fl, meters, a.home, inputs[0], H, W, a.stride, a.max_depth, a.scale_mult)
    basis, vflip, cpsnr = gs_optim.auto_convention(g0, [rgbs[0]], [p0], fov=a.fov, size=min(a.size, 256))
    print(f"[oracle] convention (single room): basis PSNR {cpsnr:.1f} dB  vflip={vflip}"
          + ("  !! LOW — inspect heldout_*.png" if cpsnr < 22 else ""))

    rows = []
    for star in extras[: a.max_rooms]:
        room = fl.panos[star]["room"]
        real = _pano_rgb(a.home, star, (H, W))
        pose = zind._pose_c2w(fl.panos[star], meters)
        preds = render_tiles(full, pose, basis, vflip, yaws, a.fov, a.size, device)   # UPRIGHT
        ps, ss_, lps, covs = [], [], [], []
        panels = []
        for y, (prgb, alpha) in zip(yaws, preds):
            gt = panoproj.e2p(real, y, 0, a.fov, (a.size, a.size))                    # real, upright
            m = alpha > a.alpha_thr
            covs.append(float(m.mean()))
            if m.sum() > 50:
                ps.append(_psnr(gt, prgb, m)); ss_.append(_ssim(gt, prgb)); lps.append(lp(gt, prgb))
            if not panels:
                holes = prgb.copy(); holes[~m] = (0, 0, 255)                          # red = unobserved
                panels = [_label(gt, "GT (real pano)"), _label(prgb, "oracle render"),
                          _label(holes, "red = holes/unobserved")]
        cv2.imwrite(str(out / f"heldout_{room}_{star[-6:]}.png"),
                    cv2.cvtColor(np.concatenate(panels, 1), cv2.COLOR_RGB2BGR))
        row = dict(room=room, held_out=star, coverage=round(float(np.mean(covs)), 3),
                   psnr=round(float(np.nanmean(ps)), 2) if ps else float("nan"),
                   ssim=round(float(np.nanmean(ss_)), 3) if ss_ else float("nan"),
                   lpips=round(float(np.nanmean(lps)), 3) if any(np.isfinite(lps)) else float("nan"))
        rows.append(row); print("  ", row)

    # summary FIRST (so a walkthrough failure can't lose the metrics)
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["room", "held_out", "coverage", "psnr", "ssim", "lpips"])
        w.writeheader(); w.writerows(rows)
    if rows:
        agg = {k: round(float(np.nanmean([r[k] for r in rows])), 3) for k in ("coverage", "psnr", "ssim", "lpips")}
        print(f"\n==== ORACLE upper bound — {a.tag} ====\n  rooms scored: {len(rows)}   mean: {agg}")
        print(f"  wrote {out}/metrics.csv, floor.ply, heldout_*.png")

    # walkthrough LAST + guarded (never lose metrics to a render hiccup)
    if a.walkthrough and len(poses) >= 2:
        try:
            from pipelines.gs_room_prototype import _lookat_c2w, gsplat_render
            wdir = out / "walkthrough"; wdir.mkdir(exist_ok=True)
            cen = np.array([p[:3, 3] for p in poses])
            path = np.concatenate([np.linspace(cen[i], cen[i + 1], a.walk_steps, endpoint=False)
                                   for i in range(len(cen) - 1)] + [cen[-1:]])
            n = 0
            for i, c in enumerate(path):
                tgt = path[min(i + 3, len(path) - 1)]
                if np.linalg.norm(tgt - c) < 1e-4:            # degenerate look-at -> skip
                    continue
                rgb, _ = gsplat_render(full, _lookat_c2w(c, tgt), basis, a.fov, a.size, device)
                if vflip:
                    rgb = rgb[::-1].copy()
                cv2.imwrite(str(wdir / f"f{n:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); n += 1
            print(f"[oracle] wrote {n} walkthrough frames -> {wdir}  (ffmpeg -framerate 8 -i f%03d.png walk.mp4)")
        except Exception as e:
            print(f"[oracle] walkthrough skipped ({e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--tag", default="oracle")
    ap.add_argument("--gs_h", type=int, default=1024, help="pano/depth height for GS init (width=2h)")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--scale_mult", type=float, default=0.0, help="0=auto (1.5*stride)")
    ap.add_argument("--max_depth", type=float, default=30.0)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--size", type=int, default=512, help="tile render size")
    ap.add_argument("--yaws", type=int, default=4, help="tiles per held-out view")
    ap.add_argument("--alpha_thr", type=float, default=0.5)
    ap.add_argument("--max_rooms", type=int, default=8)
    ap.add_argument("--walkthrough", action="store_true")
    ap.add_argument("--walk_steps", type=int, default=8)
    a = ap.parse_args()
    main(a)

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


def _pose_c2w_fixed(info, S):
    """Camera-to-world consistent with the pano-IMAGE convention (door_azimuth). ZInD's image
    azimuth relates to the floor-plan azimuth by  phi = -psi - rot_deg  (a REFLECTION), so the
    proper-rotation `zind._pose_c2w` places every room MIRRORED (windows/doors on the wrong side,
    only hidden because the render's vflip compensated). This adds the reflection so the world
    geometry — and the .ply — is correct: verified |shared-door disagreement| = 0.000 m."""
    rot = np.radians(info["rot_deg"]); C, Sr = np.cos(rot), np.sin(rot)
    R = np.array([[-C, 0.0, -Sr], [0.0, 1.0, 0.0], [-Sr, 0.0, C]])   # Ry(-rot) @ diag(-1,1,1)
    pos = np.asarray(info["pos"], float)
    T = np.eye(4); T[:3, :3] = R
    T[:3, 3] = [pos[0] * S, float(info.get("cam_h_m") or 0.0), pos[1] * S]
    return T


def _pose_c2w_render(info, S):
    """PROPER (det=+1) camera at the pano's position, forward = Ry(-rot_deg). gsplat can't render
    through the reflected build-pose, so we render the (correct-world) gaussians from this proper
    camera and flip the OUTPUT to match the pano's reflected imaging (see calibrate_render)."""
    rot = np.radians(info["rot_deg"]); C, Sr = np.cos(rot), np.sin(rot)
    R = np.array([[C, 0.0, -Sr], [0.0, 1.0, 0.0], [Sr, 0.0, C]])     # Ry(-rot), proper
    pos = np.asarray(info["pos"], float)
    T = np.eye(4); T[:3, :3] = R
    T[:3, 3] = [pos[0] * S, float(info.get("cam_h_m") or 0.0), pos[1] * S]
    return T


def calibrate_render(g, rgb, pose_proper, fov, size, device):
    """Find the (hflip, vflip) that make a proper-camera render match the real pano. hflip is
    expected (pano imaging is left-right reflected); vflip covers gsplat's y-down convention."""
    from pipelines.gs_room_prototype import gsplat_render
    gt = panoproj.e2p(rgb, 0, 0, fov, (size, size)).astype(np.float32)
    r0, _ = gsplat_render(g, pose_proper, np.eye(3, dtype=np.float32), fov, size, device)
    best = (1e18, False, False)
    for hf in (False, True):
        for vf in (False, True):
            r = r0[:, ::-1] if hf else r0
            r = r[::-1] if vf else r
            m = float(np.mean((r.astype(np.float32) - gt) ** 2))
            if m < best[0]:
                best = (m, hf, vf)
    return best[1], best[2], 99.0 if best[0] < 1e-6 else 10 * np.log10(255.0 ** 2 / best[0])


def _pano_rgb(home, stem, hw):
    p = Path(home) / "panos" / f"{stem}.jpg"
    im = cv2.imread(str(p))
    if im is None:
        return None
    return cv2.cvtColor(cv2.resize(im, (hw[1], hw[0])), cv2.COLOR_BGR2RGB)


def _load_pager_depth(home, stem, H, W, sub):
    p = Path(home) / sub / f"{stem}.npy"
    if not p.exists():
        return None
    d = np.squeeze(np.load(p).astype(np.float32))
    if d.ndim == 3:
        d = d[..., 0] if d.shape[-1] == 1 else d[0]
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    return d


def fuse_layout_pager(layout, pager, margin=0.15):
    """Clean layout walls + PaGeR where furniture sticks out IN FRONT of the wall. PaGeR is
    scale/shift-aligned to the layout (robust affine fit), then used only where it reads
    meaningfully closer than the wall -> objects appear, walls stay exact, doorway holes stay holes."""
    v = (layout > 0.1) & (pager > 0.1) & np.isfinite(pager)
    if v.sum() < 200:
        return layout
    # robust affine fit  pager -> layout  (a few IRLS steps to shrug off outliers)
    x, y = pager[v], layout[v]; a, b = 1.0, 0.0
    for _ in range(3):
        r = np.abs(a * x + b - y); w = 1.0 / (r + 0.1)
        A = np.stack([x * w, w], 1)
        sol, *_ = np.linalg.lstsq(A, y * w, rcond=None); a, b = float(sol[0]), float(sol[1])
    pa = a * pager + b
    fused = layout.copy()
    obj = v & (pa < layout - margin) & (pa > 0.1)
    fused[obj] = pa[obj]
    return fused


def _cull_depth_edges(depth, rel=0.15):
    """Zero out depth at large discontinuities (wall<->ceiling/floor seams, opening edges) so we
    don't splat 'rings' of points across those jumps. Zeroed pixels are skipped by backprojection."""
    d = depth.copy()
    gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
    gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
    thr = np.maximum(0.25, rel * d)                       # >=0.25 m or 15% of range
    d[(gx > thr) | (gy > thr)] = 0.0
    return d


def _thin_poles(g, rng, min_keep=0.12):
    """Equirect over-samples the poles (ceiling/floor) -> concentric point 'rings'. Keep each
    point with prob ~ cos(elevation) so the cloud is roughly uniform on surfaces (walls kept)."""
    xyz = g["xyz"]; r = np.linalg.norm(xyz, axis=1) + 1e-9
    cos_el = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2) / r        # 1 at horizon, 0 at poles
    keep = rng.random(len(xyz)) < np.clip(cos_el, min_keep, 1.0)
    return {k: v[keep] for k, v in g.items()}


def room_gaussians(fl, meters, home, stem, H, W, stride, max_depth, scale_mult, cull=True,
                   mask_doors=True, thin_poles=True, carve_doors=False, fuse_pager=True,
                   pager_sub="pager_depth/depth_meters"):
    depth = layout_depth.render_layout_depth(fl, stem, H, W, max_depth=max_depth,
                                             mask_doors=mask_doors, carve_doors=carve_doors)
    if fuse_pager:                                    # add furniture from PaGeR where it exists
        pg = _load_pager_depth(home, stem, H, W, pager_sub)
        if pg is not None:
            depth = fuse_layout_pager(depth, pg)
    if cull:
        depth = _cull_depth_edges(depth)
    rgb = _pano_rgb(home, stem, (H, W))
    if rgb is None:
        return None, None
    pose = _pose_c2w_fixed(fl.panos[stem], meters)
    g = build_room_gaussians(rgb, depth, pose, stride=stride, max_depth=max_depth, scale_mult=scale_mult)
    if thin_poles:
        g = _thin_poles(g, np.random.default_rng(0))
    return g, pose


def build_floor(fl, meters, home, stems, H, W, stride, max_depth, scale_mult,
                carve_doors=False, fuse_pager=True, pager_sub="pager_depth/depth_meters"):
    gs, rgbs, poses, npg = [], [], [], 0
    for s in stems:
        g, pose = room_gaussians(fl, meters, home, s, H, W, stride, max_depth, scale_mult,
                                 carve_doors=carve_doors, fuse_pager=fuse_pager, pager_sub=pager_sub)
        if g is None:
            continue
        gs.append(g); poses.append(pose)
        rgbs.append(_pano_rgb(home, s, (H, W)))
        if fuse_pager and _load_pager_depth(home, s, H, W, pager_sub) is not None:
            npg += 1
    if fuse_pager:
        print(f"[oracle] PaGeR depth fused for {npg}/{len(stems)} panos (furniture); rest layout-only")
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
def render_tiles(g, pose_proper, hflip, vflip, yaws, fov, size, device):
    """Perspective tiles from a PROPER camera at pose_proper (yaw sweep), flipped to match the
    pano's imaging. basis=identity (the proper pose already carries the rotation)."""
    from pipelines.gs_room_prototype import gsplat_render
    I = np.eye(3, dtype=np.float32)
    out = []
    for y in yaws:
        c2w = pose_proper.copy()
        c2w[:3, :3] = pose_proper[:3, :3] @ gs_optim._Ry(np.radians(-y))
        rgb, alpha = gsplat_render(g, c2w, I, fov, size, device)
        if hflip:
            rgb, alpha = rgb[:, ::-1].copy(), alpha[:, ::-1].copy()
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

    # DEFAULT: reconstruct the full floor from ALL panos.  --eval: hold one pano per room out.
    by_room = {}
    for s in panos:
        by_room.setdefault(fl.panos[s]["room"], []).append(s)
    if a.eval:
        inputs = [ss[0] for ss in by_room.values()]            # one primary pano per room
        extras = [s for ss in by_room.values() for s in ss[1:]]
        print(f"[oracle] EVAL: floor from 1 pano/room ({len(inputs)}); {len(extras)} held-out views")
    else:
        inputs, extras = list(panos), []
        print(f"[oracle] FULL floor from all {len(inputs)} panos")

    full, rgbs, poses = build_floor(fl, meters, a.home, inputs, H, W, a.stride, a.max_depth, a.scale_mult,
                                    carve_doors=a.carve_doors, fuse_pager=a.fuse_pager, pager_sub=a.pager_sub)
    if full is None:
        raise SystemExit("no panos could be loaded — check --home / that GT panos exist there")
    from sparsepano.gs import gsplat_init as gi
    gi.write_point_ply(str(out / "floor.ply"), full)
    print(f"[oracle] floor: {len(full['xyz']):,} gaussians -> {out}/floor.ply")

    # render calibration: proper camera + (hflip,vflip) that matches the real pano (solid room).
    g0, _ = room_gaussians(fl, meters, a.home, inputs[0], H, W, a.stride, a.max_depth, a.scale_mult,
                           mask_doors=False, thin_poles=False, fuse_pager=False)
    pose0 = _pose_c2w_render(fl.panos[inputs[0]], meters)
    hflip, vflip, cpsnr = calibrate_render(g0, rgbs[0], pose0, a.fov, min(a.size, 256), device)
    print(f"[oracle] render calib: hflip={hflip} vflip={vflip}  match PSNR {cpsnr:.1f} dB"
          + ("  !! LOW — inspect renders" if cpsnr < 18 else ""))

    rows = []
    if not a.eval:
        # full-floor mode: a forward "real vs render" panel from a few room cameras
        for s in inputs[: a.max_rooms]:
            real = _pano_rgb(a.home, s, (H, W))
            pose = _pose_c2w_render(fl.panos[s], meters)
            prgb, alpha = render_tiles(full, pose, hflip, vflip, [0.0], a.fov, a.size, device)[0]
            gt = panoproj.e2p(real, 0, 0, a.fov, (a.size, a.size))
            cv2.imwrite(str(out / f"view_{fl.panos[s]['room']}_{s[-6:]}.png"),
                        cv2.cvtColor(np.concatenate([_label(gt, "real pano"),
                                     _label(prgb, "floor render")], 1), cv2.COLOR_RGB2BGR))
        print(f"[oracle] wrote {min(len(inputs), a.max_rooms)} view_*.png (real | render)")
    else:
        for star in extras[: a.max_rooms]:
            room = fl.panos[star]["room"]
            real = _pano_rgb(a.home, star, (H, W))
            pose = _pose_c2w_render(fl.panos[star], meters)
            preds = render_tiles(full, pose, hflip, vflip, yaws, a.fov, a.size, device)
            ps, ss_, lps, covs, panels = [], [], [], [], []
            for y, (prgb, alpha) in zip(yaws, preds):
                gt = panoproj.e2p(real, y, 0, a.fov, (a.size, a.size))
                m = alpha > a.alpha_thr
                covs.append(float(m.mean()))
                if m.sum() > 50:
                    ps.append(_psnr(gt, prgb, m)); ss_.append(_ssim(gt, prgb)); lps.append(lp(gt, prgb))
                if not panels:
                    holes = prgb.copy(); holes[~m] = (0, 0, 255)
                    panels = [_label(gt, "GT (real pano)"), _label(prgb, "oracle render"),
                              _label(holes, "red = holes/unobserved")]
            cv2.imwrite(str(out / f"heldout_{room}_{star[-6:]}.png"),
                        cv2.cvtColor(np.concatenate(panels, 1), cv2.COLOR_RGB2BGR))
            row = dict(room=room, held_out=star, coverage=round(float(np.mean(covs)), 3),
                       psnr=round(float(np.nanmean(ps)), 2) if ps else float("nan"),
                       ssim=round(float(np.nanmean(ss_)), 3) if ss_ else float("nan"),
                       lpips=round(float(np.nanmean(lps)), 3) if any(np.isfinite(lps)) else float("nan"))
            rows.append(row); print("  ", row)
        with open(out / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                               ["room", "held_out", "coverage", "psnr", "ssim", "lpips"])
            w.writeheader(); w.writerows(rows)
        if rows:
            agg = {k: round(float(np.nanmean([r[k] for r in rows])), 3) for k in ("coverage", "psnr", "ssim", "lpips")}
            print(f"\n==== ORACLE held-out — {a.tag} ====\n  rooms scored: {len(rows)}   mean: {agg}")

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
                rgb, _ = gsplat_render(full, _lookat_c2w(c, tgt), np.eye(3, dtype=np.float32),
                                       a.fov, a.size, device)
                if vflip:
                    rgb = rgb[::-1].copy()
                cv2.imwrite(str(wdir / f"f{n:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); n += 1
            print(f"[oracle] wrote {n} walkthrough frames -> {wdir}")
            # encode to mp4 if ffmpeg is available; else leave frames + the manual command
            import shutil, subprocess
            mp4 = out / "walkthrough.mp4"
            if shutil.which("ffmpeg"):
                subprocess.run(["ffmpeg", "-y", "-framerate", "8", "-i", str(wdir / "f%03d.png"),
                                "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                str(mp4)], check=True, capture_output=True)
                print(f"[oracle] encoded {mp4}")
            else:
                print(f"[oracle] ffmpeg not found; encode with: ffmpeg -framerate 8 -i {wdir}/f%03d.png {mp4}")
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
    ap.add_argument("--eval", action="store_true",
                    help="held-out novel-view metrics (1 pano/room in, rest scored). Default: full floor from all panos.")
    ap.add_argument("--no_fuse", dest="fuse_pager", action="store_false",
                    help="disable PaGeR fusion (layout-only geometry, walls but no furniture)")
    ap.add_argument("--pager_sub", default="pager_depth/depth_meters", help="per-home PaGeR depth subdir")
    ap.add_argument("--carve_doors", action="store_true",
                    help="also carve DOORS into holes (default: only true openings; keeps closed doors solid)")
    ap.set_defaults(fuse_pager=True)
    ap.add_argument("--walkthrough", action="store_true")
    ap.add_argument("--walk_steps", type=int, default=8)
    a = ap.parse_args()
    main(a)

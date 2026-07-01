"""
Differentiable Gaussian-splatting optimization (GPU) — reference implementation.

NOT runnable in the CPU sandbox (needs torch + gsplat). Supervises the per-pano Gaussian
init (src/gsplat_init) by rendering PERSPECTIVE TILES sampled from the panoramas (gsplat is
pinhole, not equirect — the 360-GS trick) and minimizing L1. Gaussians live in the WORLD
frame; each pano contributes tiles at its own camera pose, so disoccluded regions get
cross-view supervision (meaningful for >=2 panos; a single pano just reproduces itself).

Camera conventions (e2p is Y-up; gsplat builds may be OpenCV/OpenGL, and an image vertical
flip may be needed) are impossible to verify on CPU, so we BRUTE-FORCE the small set of
(rotation-basis x image-vflip) combos and auto-pick the one whose render of the INIT best
matches the e2p tiles (`auto_convention`). Run reliably without guessing flags.

Install (Windows, prebuilt, no compiler): pick the wheel matching torch+cuda, e.g.
  pip install gsplat==1.5.2+pt20cu118 --index-url https://pypi.org/simple \
      --extra-index-url https://docs.gsplat.studio/whl
"""
import numpy as np

# Candidate camera bases (proper rotations: even number of axis sign-flips, det=+1).
_BASES = [np.diag(v).astype(np.float32) for v in
          [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]


def ensure_msvc_on_path():
    """Windows: gsplat JIT-compiles CUDA kernels and needs MSVC `cl.exe` on PATH. If it's
    missing, locate the VS C++ toolchain via vswhere, run vcvars64.bat, and import its
    environment. No-op off Windows or if cl is already available / using a prebuilt wheel."""
    import os, sys, shutil, subprocess
    if sys.platform != "win32" or shutil.which("cl"):
        return
    pf = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        return                      # prebuilt wheel may not need cl; let gsplat speak if it does
    try:
        install = subprocess.check_output(
            [vswhere, "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"], text=True).strip()
        vcvars = os.path.join(install, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if not os.path.exists(vcvars):
            return
        out = subprocess.check_output(f'cmd.exe /s /c ""{vcvars}" >NUL && set"', text=True)
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1); os.environ[k] = v
        if shutil.which("cl"):
            print("[gs_optim] activated MSVC environment (cl.exe now on PATH)")
    except Exception:
        return


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def _K(fov, size):
    f = 0.5 * size / np.tan(np.radians(fov) / 2)
    return np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]], np.float32)


def _tile_views(pose_c2w, yaws, pitches, basis):
    Rcw, t = pose_c2w[:3, :3], pose_c2w[:3, 3]
    views, dirs = [], []
    for yaw in yaws:
        for pitch in pitches:
            Rc2w = Rcw @ (_Ry(np.radians(yaw)) @ _Rx(np.radians(pitch))) @ basis
            T = np.eye(4, dtype=np.float32); T[:3, :3] = Rc2w; T[:3, 3] = t
            views.append(np.linalg.inv(T)); dirs.append((yaw, pitch))
    return np.stack(views), dirs


def _supervision(panos, poses_c2w, yaws, pitches, fov, size, basis, vflip):
    from src import panoproj
    imgs, viewmats = [], []
    V_list = [_tile_views(pose, yaws, pitches, basis) for pose in poses_c2w]
    for rgb, (V, dirs) in zip(panos, V_list):
        for (yaw, pitch), vm in zip(dirs, V):
            tile = panoproj.e2p(rgb, yaw, pitch, fov, (size, size)).astype(np.float32) / 255.0
            if vflip:
                tile = tile[::-1].copy()
            imgs.append(tile); viewmats.append(vm)
    return np.stack(imgs), np.stack(viewmats)


def _to_raw(g, device):
    import torch
    def t(x): return torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
    def logit(x): x = np.clip(x, 1e-4, 1 - 1e-4); return np.log(x / (1 - x))
    return {"means": t(g["xyz"]).requires_grad_(True),
            "quats": t(g["rot"]).requires_grad_(True),
            "log_scales": t(np.log(np.clip(g["scale"], 1e-6, None))).requires_grad_(True),
            "logit_opac": t(logit(g["opacity"])).requires_grad_(True),
            "raw_colors": t(logit(np.clip(g["rgb"], 1e-4, 1 - 1e-4))).requires_grad_(True)}


def _activated(raw):
    import torch
    return (raw["means"], torch.nn.functional.normalize(raw["quats"], dim=-1),
            torch.exp(raw["log_scales"]), torch.sigmoid(raw["logit_opac"]),
            torch.sigmoid(raw["raw_colors"]))


def _render(raw, viewmats, K, size, device):
    import torch
    from gsplat import rasterization
    means, quats, scales, opac, colors = _activated(raw)
    Ks = torch.tensor(K, device=device).float()[None].repeat(len(viewmats), 1, 1)
    vm = torch.tensor(np.asarray(viewmats), device=device).float()
    out, _, _ = rasterization(means, quats, scales, opac, colors, vm, Ks,
                              width=size, height=size, render_mode="RGB")
    return out


def auto_convention(g, panos, poses_c2w, fov=90, size=256, device="cuda"):
    """Try every (basis x vflip) combo; return (basis, vflip, psnr) whose render of the
    INIT best matches the e2p tiles. The init already reproduces the input, so the correct
    convention scores high; wrong ones score low."""
    import torch
    yaws = list(range(0, 360, 90)); pitches = [0.0]; K = _K(fov, size)
    raw = _to_raw(g, device); best = None
    for bi, basis in enumerate(_BASES):
        for vflip in (False, True):
            imgs, vm = _supervision(panos, poses_c2w, yaws, pitches, fov, size, basis, vflip)
            with torch.no_grad():
                ren = _render(raw, vm, K, size, device).clamp(0, 1).cpu().numpy()
            mse = float(np.mean((ren - imgs) ** 2))
            psnr = 99.0 if mse < 1e-8 else 10 * np.log10(1.0 / mse)
            if best is None or psnr > best[2]:
                best = (basis, vflip, psnr, bi)
    msg = ("" if best[2] > 22 else
           "  -- moderate: either camera convention OR scene overlap (multi-room bleed)")
    print(f"[gs_optim] auto-convention: basis#{best[3]} vflip={best[1]}  PSNR {best[2]:.1f} dB{msg}")
    return best[0], best[1], best[2]


def convention_check(g, panos, poses_c2w, fov=90, size=256, device="cuda", **_):
    return auto_convention(g, panos, poses_c2w, fov, size, device)[2]


def optimize(g, panos, poses_c2w, iters=3000, lr=1e-2, fov=90, size=256,
             n_yaw=8, pitches=(-20.0, 0.0, 20.0), device="cuda", log_every=200, **_):
    """Refine Gaussians against perspective tiles from all panos. Returns updated g dict."""
    import torch
    ensure_msvc_on_path()
    basis, vflip, psnr = auto_convention(g, panos, poses_c2w, fov, size, device)
    if psnr < 18:
        print("[gs_optim] NOTE: PSNR < 18 dB. If a single room alone scores ~30 dB, the drop "
              "is multi-room BLEED (overlapping clouds / weak occlusion), not a convention bug "
              "-- try a larger --gscale; inspect the _tiles.png to confirm.")
    yaws = list(np.linspace(0, 360, n_yaw, endpoint=False))
    imgs, viewmats = _supervision(panos, poses_c2w, yaws, list(pitches), fov, size, basis, vflip)
    imgs_t = torch.tensor(imgs, device=device)
    K = _K(fov, size); Ks = torch.tensor(K, device=device).float()[None]
    vm_t = torch.tensor(viewmats, device=device).float()
    raw = _to_raw(g, device); opt = torch.optim.Adam(list(raw.values()), lr=lr)
    from gsplat import rasterization
    M = len(imgs)
    for it in range(iters):
        j = np.random.randint(M)
        means, quats, scales, opac, colors = _activated(raw)
        out, _, _ = rasterization(means, quats, scales, opac, colors, vm_t[j:j+1], Ks,
                                  width=size, height=size, render_mode="RGB")
        loss = (out[0] - imgs_t[j]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % log_every == 0:
            print(f"  gs-opt it {it}/{iters}  L1 {loss.item():.4f}")
    means, quats, scales, opac, colors = _activated(raw)
    return dict(xyz=means.detach().cpu().numpy(), rot=quats.detach().cpu().numpy(),
                scale=scales.detach().cpu().numpy(),
                opacity=opac.detach().cpu().numpy(), rgb=colors.detach().cpu().numpy())


def save_debug_tiles(g, panos, poses_c2w, out_path, fov=90, size=256, device="cuda"):
    """Render the INIT at each pano's tiles (auto-convention) and save a mosaic
    [target | render] per tile, plus per-pano PSNR — to SEE why a pose misaligns."""
    import torch, cv2
    basis, vflip, _ = auto_convention(g, panos, poses_c2w, fov, size, device)
    yaws = [0, 90, 180, 270]; pitches = [0.0]; tpp = len(yaws) * len(pitches)
    raw = _to_raw(g, device); K = _K(fov, size)
    imgs, vm = _supervision(panos, poses_c2w, yaws, pitches, fov, size, basis, vflip)
    with torch.no_grad():
        ren = _render(raw, vm, K, size, device).clamp(0, 1).cpu().numpy()
    for pi in range(len(panos)):
        seg_i = imgs[pi*tpp:(pi+1)*tpp]; seg_r = ren[pi*tpp:(pi+1)*tpp]
        mse = float(np.mean((seg_i - seg_r) ** 2))
        print(f"  [debug] pano {pi}: tile PSNR {(10*np.log10(1/mse) if mse>1e-8 else 99):.1f} dB")
    rows = [np.concatenate([imgs[i], ren[i]], axis=1) for i in range(len(imgs))]
    grid = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  [debug] saved tiles (target | render) -> {out_path}")

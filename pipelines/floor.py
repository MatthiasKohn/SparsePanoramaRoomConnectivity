"""
Floor reconstruction pipeline with SWAPPABLE building blocks (the substitution study).

One command, one code path, one eval -> change exactly one block via a flag and the results are
comparable. Every run appends a row to a master results.csv (keyed by the config) so you get the
comparison table + the quality-vs-pose-error curve.

    python -m pipelines.floor --home $ZIND_ROOT/0000 --floor floor_01 \
        --pose_model gt   --depth_model fused        # the oracle (upper bound)
    python -m pipelines.floor ... --pose_model noise --noise_deg 10 --noise_m 0.3   # pose sensitivity
    python -m pipelines.floor ... --depth_model pager                               # depth swap
    ... --visuals   # also build the full floor + ply + room-aware walkthrough.mp4

Blocks (providers): sparsepano/providers/{poses,depth,connectivity,completion}.py
Metrics: held-out novel-view PSNR/SSIM/LPIPS + coverage (reconstruction quality) and Umeyama
camera RMSE + rotation error (pose quality).
"""
import os, csv, argparse, shutil
from pathlib import Path
import numpy as np
import cv2

from sparsepano import config
from sparsepano.datasets import zind_floor
from sparsepano.geometry import panoproj
from sparsepano.providers import poses as PRO_POSE, depth as PRO_DEPTH
from sparsepano.providers import connectivity as PRO_CONN, completion as PRO_COMP


# gs_optim/gsplat_init/gs_room_prototype pull in torch+gsplat (heavy, need CUDA). Import them lazily
# so `--metrics_only` (pose error + door consistency) stays pure-numpy and light enough for a login node.
def _load_gs():
    global gs_optim, gi, build_room_gaussians, merge, gsplat_render, _lookat_c2w
    from sparsepano.gs import gs_optim, gsplat_init as gi
    from pipelines.gs_room_prototype import build_room_gaussians, merge, gsplat_render, _lookat_c2w


# ----------------------------------------------------------------- small helpers
def _pano_rgb(home, stem, hw):
    im = cv2.imread(str(Path(home) / "panos" / f"{stem}.jpg"))
    return None if im is None else cv2.cvtColor(cv2.resize(im, (hw[1], hw[0])), cv2.COLOR_BGR2RGB)


def _cull_depth_edges(d, rel=0.15):
    dd = d.copy()
    gx = np.abs(np.diff(dd, axis=1, prepend=dd[:, :1])); gy = np.abs(np.diff(dd, axis=0, prepend=dd[:1, :]))
    dd[(gx > np.maximum(0.25, rel * dd)) | (gy > np.maximum(0.25, rel * dd))] = 0.0
    return dd


def _thin_poles(g, rng, min_keep=0.12):
    xyz = g["xyz"]; r = np.linalg.norm(xyz, axis=1) + 1e-9
    keep = rng.random(len(xyz)) < np.clip(np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2) / r, min_keep, 1.0)
    return {k: v[keep] for k, v in g.items()}


def _flip_x180(g):
    xyz = g["xyz"].copy(); xyz[:, 1] *= -1; xyz[:, 2] *= -1
    out = {**g, "xyz": xyz}
    if "rot" in g and g["rot"].shape[1] == 4:
        w, x, y, z = g["rot"].T; out["rot"] = np.stack([-x, w, -z, y], 1).astype(np.float32)
    return out


def _smooth_tour(centers, steps=18):
    pts = [np.asarray(c, float) for c in centers]
    order, used = [0], {0}
    for _ in range(len(pts) - 1):
        last = order[-1]
        order.append(min((np.linalg.norm(pts[i] - pts[last]), i) for i in range(len(pts)) if i not in used)[1])
        used.add(order[-1])
    P = np.array([pts[i] for i in order]); P[:, 1] = float(np.median(P[:, 1]))
    def cr(p0, p1, p2, p3, t):
        t2, t3 = t * t, t * t * t
        return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
    ext = np.vstack([P[0], P, P[-1]]); path = []
    for i in range(1, len(ext) - 2):
        for s in range(steps):
            path.append(cr(ext[i - 1], ext[i], ext[i + 1], ext[i + 2], s / steps))
    path.append(P[-1]); return np.array(path)


def _label(img, text):
    img = img.copy(); cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA); return img


# ----------------------------------------------------------------- metrics
def _psnr(a, b, m):
    a, b = a[m].astype(np.float32), b[m].astype(np.float32)
    if a.size == 0:
        return float("nan")
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-6 else float(10 * np.log10(255.0 ** 2 / mse))


def _ssim(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32); gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2; k, s = (11, 11), 1.5
    ma, mb = cv2.GaussianBlur(ga, k, s), cv2.GaussianBlur(gb, k, s)
    va = cv2.GaussianBlur(ga * ga, k, s) - ma ** 2; vb = cv2.GaussianBlur(gb * gb, k, s) - mb ** 2
    vab = cv2.GaussianBlur(ga * gb, k, s) - ma * mb
    return float(np.clip(((2 * ma * mb + C1) * (2 * vab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2) + 1e-12), -1, 1).mean())


class _LPIPS:
    def __init__(self, device):
        self.fn = None
        try:
            import lpips, torch
            self.fn = lpips.LPIPS(net="alex").to(device).eval(); self.t = torch; self.dev = device
        except Exception as e:
            print(f"[floor] LPIPS unavailable ({e})")

    def __call__(self, a, b):
        if self.fn is None:
            return float("nan")
        to = lambda x: self.t.tensor(x.transpose(2, 0, 1)[None] / 127.5 - 1, dtype=self.t.float32, device=self.dev)
        with self.t.no_grad():
            return float(self.fn(to(a), to(b)).item())


def pose_errors(gt, est, stems):
    """Umeyama-aligned camera-centre RMSE (m) and mean rotation error (deg) over `stems`."""
    G = np.array([gt[s][:3, 3] for s in stems]); E = np.array([est[s][:3, 3] for s in stems])
    rot = float(np.mean([np.degrees(np.arccos(np.clip((np.trace(gt[s][:3, :3].T @ est[s][:3, :3]) - 1) / 2, -1, 1))) for s in stems]))
    if len(G) < 3:
        return float("nan"), rot
    Gc, Ec = G - G.mean(0), E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec.T @ Gc); R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = U @ Vt
    sc = S.sum() / (Ec ** 2).sum()
    return float(np.sqrt(np.mean(np.sum((sc * (Ec @ R) + G.mean(0) - G) ** 2, 1)))), rot


def align_poses_to_gt(build_poses, render_poses, gt_poses, stems):
    """Align estimated poses to GT by a global Sim(3) (reflection allowed) fit on camera centres.

    Real pose methods (SALVe) output poses in their OWN world frame (different global rotation, scale,
    and often a mirror), because pose is only recoverable up to a global similarity. Before measuring
    rotation error or door consistency (which assume est and GT share a frame), we fit one global
    Sim(3) mapping est centres -> GT centres and apply it to every estimated pose. Reflection is
    allowed (SALVe's frame is mirrored: its angle = -ours). Returns (build', render')."""
    S = [s for s in stems if s in build_poses and s in gt_poses]
    if len(S) < 3:
        return build_poses, render_poses
    E = np.array([build_poses[s][:3, 3] for s in S]); G = np.array([gt_poses[s][:3, 3] for s in S])
    Em, Gm = E.mean(0), G.mean(0); Ec, Gc = E - Em, G - Gm
    U, Dsv, Vt = np.linalg.svd(Ec.T @ Gc)
    R = (U @ Vt).T                                   # reflection allowed (SALVe's frame is mirrored)
    sc = Dsv.sum() / max(np.sum(Ec ** 2), 1e-12)
    # one uniform similarity T = [[sc*R, t],[0,1]] mapping the est frame onto GT. door_gap is invariant
    # to any uniform T, so left-multiplying every est pose by T makes est & GT share a frame.
    T = np.eye(4); T[:3, :3] = sc * R; T[:3, 3] = Gm - sc * R @ Em
    return ({s: T @ P for s, P in build_poses.items()},
            {s: T @ P for s, P in render_poses.items()})


def door_consistency(fl, rooms_map, adj, build_poses, gt_poses, meters):
    """Cross-room / global-consistency metric (needs no rendering).

    A shared doorway is ONE physical world point. Each adjacent room places it via its own
    (estimated) pose; under GT the two placements coincide (~0 m), under pose error they diverge.
    For a room, its rigid placement error is the delta between its build pano's estimated and GT
    pose: D = P_est @ inv(P_gt). We carry the GT door midpoint by each room's D and measure how far
    apart the two rooms now think the door is. Reflection convention cancels in the delta.

    Returns door_gap_m (mean), door_gap_max_m, n_doors. This is sensitive to pose error exactly
    where the within-room PSNR curve is flat -> it isolates whether the assembled floor stays
    globally consistent (rooms actually meet at their shared doors)."""
    build_pano = {r: ss[0] for r, ss in rooms_map.items()}
    gaps, seen = [], set()
    for r, neigh in adj.items():
        for s in neigh:
            key = tuple(sorted((r, s)))
            if key in seen or r not in build_pano or s not in build_pano:
                continue
            seen.add(key)
            # door LOCATION from whichever pano-pair actually shares it (geometry only)...
            mid = None
            for a in rooms_map[r]:
                for b in rooms_map[s]:
                    mid = fl.shared_door(a, b, tol=0.25)
                    if mid is not None:
                        break
                if mid is not None:
                    break
            if mid is None:
                continue
            # ...MOTION from each room's build pano (the pano whose pose places that room's cloud).
            p = np.array([mid[0] * meters, 0.0, mid[1] * meters, 1.0])
            dR = build_poses[build_pano[r]] @ np.linalg.inv(gt_poses[build_pano[r]])
            dS = build_poses[build_pano[s]] @ np.linalg.inv(gt_poses[build_pano[s]])
            mR, mS = dR @ p, dS @ p
            gaps.append(float(np.linalg.norm((mR - mS)[[0, 2]])))
    if not gaps:
        return dict(door_gap_m=float("nan"), door_gap_max_m=float("nan"), n_doors=0)
    return dict(door_gap_m=round(float(np.mean(gaps)), 3),
                door_gap_max_m=round(float(max(gaps)), 3), n_doors=len(gaps))


def append_results(master, row):
    """Schema-robust append: union the columns of the existing file with this row and rewrite, so
    adding a new metric column never misaligns older rows."""
    rows, fields = [], list(row.keys())
    if master.exists():
        with open(master, newline="") as f:
            rows = list(csv.DictReader(f))
        seen = set()
        fields = [c for c in (list(rows[0].keys()) if rows else []) + list(row.keys())
                  if not (c in seen or seen.add(c))]
    rows.append(row)
    with open(master, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rr in rows:
            w.writerow(rr)


# ----------------------------------------------------------------- build + render
def build_floor(fl, meters, home, rooms_map, build_poses, H, W, a):
    """Per pano: depth (provider) -> gaussians at the (est) build pose, tagged with a room index."""
    gs, ridx, rgbs, used = [], [], [], []
    room_ids = list(rooms_map)
    for i, r in enumerate(room_ids):
        for s in rooms_map[r]:
            depth = _cull_depth_edges(PRO_DEPTH.get_depth(fl, home, s, H, W, model=a.depth_model,
                                     max_depth=a.max_depth, carve_doors=a.carve_doors))
            rgb = _pano_rgb(home, s, (H, W))
            if rgb is None:
                continue
            g = build_room_gaussians(rgb, depth, build_poses[s], stride=a.stride,
                                     max_depth=a.max_depth, scale_mult=a.scale_mult)
            g = _thin_poles(g, np.random.default_rng(0))
            gs.append(g); ridx.append(np.full(len(g["xyz"]), i, np.int32)); rgbs.append(rgb); used.append(s)
    if not gs:
        return None, None, room_ids, [], []
    return merge(gs), np.concatenate(ridx), room_ids, rgbs, used


def _subset(full, ridx, keep):
    m = np.isin(ridx, list(keep))
    return {k: v[m] for k, v in full.items()}


def calibrate_render(g, rgb, pose_render, fov, size, device):
    gt = panoproj.e2p(rgb, 0, 0, fov, (size, size)).astype(np.float32)
    r0, _ = gsplat_render(g, pose_render, np.eye(3, dtype=np.float32), fov, size, device)
    best = (1e18, False, False)
    for hf in (False, True):
        for vf in (False, True):
            r = r0[:, ::-1] if hf else r0; r = r[::-1] if vf else r
            m = float(np.mean((r.astype(np.float32) - gt) ** 2))
            if m < best[0]:
                best = (m, hf, vf)
    return best[1], best[2]


def render_view(g, pose_render, hflip, vflip, yaw, fov, size, device):
    c2w = pose_render.copy(); c2w[:3, :3] = pose_render[:3, :3] @ gs_optim._Ry(np.radians(-yaw))
    rgb, alpha = gsplat_render(g, c2w, np.eye(3, dtype=np.float32), fov, size, device)
    if hflip:
        rgb, alpha = rgb[:, ::-1].copy(), alpha[:, ::-1].copy()
    if vflip:
        rgb, alpha = rgb[::-1].copy(), alpha[::-1].copy()
    return rgb, alpha


def optimize_floor(g, fl, meters, home, stems, render_poses, hflip, vflip, fov, size, iters, lr, device):
    import torch
    from gsplat import rasterization
    K = torch.tensor(gs_optim._K(fov, size), device=device).float()[None]
    sup = []
    for s in stems:
        rgb = _pano_rgb(home, s, (max(size, 512), max(size, 512) * 2))
        if rgb is None:
            continue
        base = render_poses[s]
        for y in (0, 90, 180, 270):
            R = base[:3, :3] @ gs_optim._Ry(np.radians(-y))
            T = np.eye(4, dtype=np.float32); T[:3, :3] = R; T[:3, 3] = base[:3, 3]
            vm = torch.tensor(np.linalg.inv(T), device=device).float()[None]
            gt = panoproj.e2p(rgb, y, 0, fov, (size, size)).astype(np.float32) / 255.0
            sup.append((vm, torch.tensor(gt, device=device)))
    if not sup:
        return g
    raw = gs_optim._to_raw(g, device); raw["means"].requires_grad_(False)
    opt = torch.optim.Adam([raw["quats"], raw["log_scales"], raw["logit_opac"], raw["raw_colors"]], lr=lr)
    rng = np.random.default_rng(0)
    for it in range(iters):
        vm, gt = sup[int(rng.integers(len(sup)))]
        means, quats, scales, opac, colors = gs_optim._activated(raw)
        out, _, _ = rasterization(means, quats, scales, opac, colors, vm, K, width=size, height=size, render_mode="RGB")
        img = out[0]
        if hflip:
            img = torch.flip(img, [1])
        if vflip:
            img = torch.flip(img, [0])
        loss = (img - gt).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, iters // 5) == 0:
            print(f"  [gs-opt] it {it}/{iters} L1 {loss.item():.4f}")
    means, quats, scales, opac, colors = gs_optim._activated(raw)
    return dict(xyz=means.detach().cpu().numpy(), rot=quats.detach().cpu().numpy(),
                scale=scales.detach().cpu().numpy(), opacity=opac.detach().cpu().numpy(),
                rgb=colors.detach().cpu().numpy(), room=g["room"])


# ----------------------------------------------------------------- main
def main(a):
    fl = zind_floor.ZindFloor(Path(a.home) / "zind_data.json", floor=a.floor)
    meters = float(fl.meters_per_coord)
    panos = [p for p in fl.panos if len(np.asarray(fl.panos[p]["verts_global"])) >= 3]
    home_id = Path(a.home).name
    tag = a.tag or f"{home_id}_{a.floor}_pose-{a.pose_model}_depth-{a.depth_model}"
    out = config.RESULTS_ROOT / "floor" / tag; out.mkdir(parents=True, exist_ok=True)
    H, W = a.gs_h, a.gs_h * 2
    print(f"[floor] {home_id}/{a.floor}: {len(panos)} panos | pose={a.pose_model} depth={a.depth_model} "
          f"conn={a.connectivity} completion={a.completion}")

    build_poses, render_poses, gt_poses = PRO_POSE.get_poses(
        fl, meters, a.pose_model, a.noise_deg, a.noise_m, a.seed, a.pose_file)
    # a pose provider may localize only a subset (e.g. SALVe) -> keep only panos it placed
    panos = [p for p in panos if p in build_poses]
    rooms_map, adj = PRO_CONN.get_rooms(fl, panos, a.connectivity)
    rooms_map = {r: [s for s in ss if s in build_poses] for r, ss in rooms_map.items()}
    rooms_map = {r: ss for r, ss in rooms_map.items() if ss}
    # real pose methods output in their own global frame (pose is defined only up to a similarity) ->
    # align to GT before measuring rotation error / door consistency. GT/noise/drift are already in
    # our frame (noise/drift ARE the error to measure), so they are left untouched.
    if a.pose_model in ("salve", "badgr", "covispose"):
        build_poses, render_poses = align_poses_to_gt(build_poses, render_poses, gt_poses, panos)
    p_rmse, p_rot = pose_errors(gt_poses, build_poses, panos)
    print(f"[floor] pose error vs GT: RMSE {p_rmse:.3f} m | rot {p_rot:.1f} deg")

    # ---------- metrics-only: pose error + door consistency, NO rendering (no gsplat needed) ----------
    if a.metrics_only:
        door = door_consistency(fl, rooms_map, adj, build_poses, gt_poses, meters)
        print(f"[floor] door consistency: gap {door['door_gap_m']} m (max {door['door_gap_max_m']}) over {door['n_doors']} shared doors")
        row = dict(home=home_id, floor=a.floor, pose_model=a.pose_model, depth_model=a.depth_model,
                   connectivity=a.connectivity, completion=a.completion,
                   noise_deg=a.noise_deg, noise_m=a.noise_m, pose_rmse_m=round(p_rmse, 3), rot_err_deg=round(p_rot, 2),
                   n_rooms=len(rooms_map), n_eval=0,
                   coverage=float("nan"), psnr=float("nan"), ssim=float("nan"), lpips=float("nan"), **door)
        append_results(config.RESULTS_ROOT / "floor" / "results.csv", row)
        print(f"[floor] (metrics_only) appended row -> {config.RESULTS_ROOT / 'floor' / 'results.csv'}")
        return

    # everything below needs rendering -> now (and only now) pull in torch + gsplat
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _load_gs()

    # convention calibration from a SOLID GT room (render convention is global)
    r0 = list(rooms_map.values())[0][0]
    d0 = PRO_DEPTH.get_depth(fl, a.home, r0, H, W, model="gt_layout", mask_doors=False)
    g0 = build_room_gaussians(_pano_rgb(a.home, r0, (H, W)), d0, build_poses[r0],
                              stride=a.stride, max_depth=a.max_depth, scale_mult=a.scale_mult)
    hflip, vflip = calibrate_render(g0, _pano_rgb(a.home, r0, (H, W)), render_poses[r0], a.fov, min(a.size, 256), device)
    lp = _LPIPS(device)

    # ---------- METRICS: held-out novel view (floor from 1 pano/room, score the rest) ----------
    inputs = {r: ss[0] for r, ss in rooms_map.items()}
    extras = [s for ss in rooms_map.values() for s in ss[1:]]
    eval_map = {r: [inputs[r]] for r in rooms_map}
    ef, eridx, eroom_ids, _, _ = build_floor(fl, meters, a.home, eval_map, build_poses, H, W, a)
    ps, ss_, lps, covs = [], [], [], []
    for star in extras[: a.max_eval]:
        room = fl.panos[star]["room"]; ri = eroom_ids.index(room)
        keep = {ri} | {eroom_ids.index(r) for r in adj.get(room, set()) if r in eroom_ids}
        sub = _subset(ef, eridx, keep)
        real = _pano_rgb(a.home, star, (H, W))
        for y in np.linspace(0, 360, a.yaws, endpoint=False):
            prgb, alpha = render_view(sub, render_poses[star], hflip, vflip, y, a.fov, a.size, device)
            gt = panoproj.e2p(real, y, 0, a.fov, (a.size, a.size)); m = alpha > a.alpha_thr
            covs.append(float(m.mean()))
            if m.sum() > 50:
                ps.append(_psnr(gt, prgb, m)); ss_.append(_ssim(gt, prgb)); lps.append(lp(gt, prgb))
    metrics = dict(coverage=round(float(np.mean(covs)), 3) if covs else float("nan"),
                   psnr=round(float(np.nanmean(ps)), 2) if ps else float("nan"),
                   ssim=round(float(np.nanmean(ss_)), 3) if ss_ else float("nan"),
                   lpips=round(float(np.nanmean(lps)), 3) if any(np.isfinite(lps)) else float("nan"))
    print(f"[floor] held-out metrics: {metrics}  (rooms={len(rooms_map)} eval_views={min(len(extras), a.max_eval)})")

    # ---------- cross-room / global-consistency metric (door agreement under the est poses) ----------
    door = door_consistency(fl, rooms_map, adj, build_poses, gt_poses, meters)
    print(f"[floor] door consistency: gap {door['door_gap_m']} m (max {door['door_gap_max_m']}) over {door['n_doors']} shared doors")

    # ---------- append one row to the master results.csv ----------
    row = dict(home=home_id, floor=a.floor, pose_model=a.pose_model, depth_model=a.depth_model,
               connectivity=a.connectivity, completion=a.completion,
               noise_deg=a.noise_deg, noise_m=a.noise_m, pose_rmse_m=round(p_rmse, 3), rot_err_deg=round(p_rot, 2),
               n_rooms=len(rooms_map), n_eval=min(len(extras), a.max_eval), **metrics, **door)
    master = config.RESULTS_ROOT / "floor" / "results.csv"
    append_results(master, row)
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
    print(f"[floor] appended row -> {master}")

    # ---------- optional VISUALS: full floor + room-aware walkthrough ----------
    if a.visuals:
        full, fridx, froom_ids, rgbs, used = build_floor(fl, meters, a.home, rooms_map, build_poses, H, W, a)
        full["room"] = fridx
        if a.optimize:
            print(f"[floor] 3DGS optimization: {a.opt_iters} iters @ {a.opt_size}px")
            full = optimize_floor(full, fl, meters, a.home, used, render_poses, hflip, vflip,
                                  a.fov, a.opt_size, a.opt_iters, a.opt_lr, device)
        if len(full["xyz"]) > a.view_points:
            idx = np.random.default_rng(0).choice(len(full["xyz"]), a.view_points, replace=False)
            light = {k: v[idx] for k, v in full.items()}
            gi.write_point_ply(str(out / "floor_light.ply"), light)
            gi.write_gs_ply(str(out / "floor_light_gs.ply"), _flip_x180(light))
        gi.write_gs_ply(str(out / "floor_gs.ply"), _flip_x180(full))
        print(f"[floor] visuals: {len(full['xyz']):,} gaussians -> floor_gs.ply (+ *_light for scp)")
        _walkthrough(full, fridx, froom_ids, adj, fl, rooms_map, render_poses, hflip, vflip, out, a, device)


def _walkthrough(full, fridx, froom_ids, adj, fl, rooms_map, render_poses, hflip, vflip, out, a, device):
    shutil.rmtree(out / "walkthrough", ignore_errors=True)
    sd = None
    if a.completion == "sd":
        try:
            sd = PRO_COMP.load_sd_inpainter(device); print("[floor] SD inpainter loaded")
        except Exception as e:
            print(f"[floor] SD unavailable ({e}); using cv2"); a.completion = "cv2"
    room_pos = {r: np.mean([render_poses[s][[0, 2], 3] for s in ss], 0) for r, ss in rooms_map.items()}
    centers = [render_poses[s][:3, 3] for ss in rooms_map.values() for s in ss]
    path = _smooth_tour(centers, steps=a.walk_steps)
    if len(path) > a.walk_frames:
        path = path[np.linspace(0, len(path) - 1, a.walk_frames).astype(int)]
    wsize = a.walk_size or a.size
    vw = cv2.VideoWriter(str(out / "walkthrough.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.walk_fps, (wsize, wsize))
    n = 0
    for i, c in enumerate(path):
        tgt = path[min(i + 6, len(path) - 1)]
        if np.linalg.norm(tgt - c) < 1e-4:
            continue
        cur = min(room_pos, key=lambda r: np.linalg.norm(room_pos[r] - c[[0, 2]]))     # room the camera is in
        keep = {froom_ids.index(cur)} | {froom_ids.index(r) for r in adj.get(cur, set()) if r in froom_ids}
        sub = _subset(full, fridx, keep)                                                # room-aware -> no see-through
        rgb, _ = gsplat_render(sub, _lookat_c2w(c, tgt), np.eye(3, dtype=np.float32), a.fov, wsize, device)
        if vflip:
            rgb = rgb[::-1].copy()
        if hflip:
            rgb = rgb[:, ::-1].copy()
        vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); n += 1
    vw.release()
    print(f"[floor] walkthrough -> {out}/walkthrough.mp4 ({n} frames, room-aware, completion='{a.completion}')")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True); ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--tag", default="")
    # --- swappable blocks ---
    ap.add_argument("--metrics_only", action="store_true",
                    help="compute pose error + door consistency only (no gsplat rendering / PSNR)")
    ap.add_argument("--pose_model", default="gt", choices=["gt", "noise", "drift", "salve", "badgr", "covispose"])
    ap.add_argument("--noise_deg", type=float, default=0.0); ap.add_argument("--noise_m", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pose_file", default="")
    ap.add_argument("--depth_model", default="fused", choices=["gt_layout", "fused", "pager", "dap"])
    ap.add_argument("--connectivity", default="gt", choices=["gt", "detected"])
    ap.add_argument("--completion", default="cv2", choices=["none", "cv2", "sd"])
    # --- geometry/render ---
    ap.add_argument("--gs_h", type=int, default=1024); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--scale_mult", type=float, default=2.5); ap.add_argument("--max_depth", type=float, default=15.0)
    ap.add_argument("--carve_doors", action="store_true")
    ap.add_argument("--fov", type=float, default=90.0); ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--yaws", type=int, default=4); ap.add_argument("--alpha_thr", type=float, default=0.5)
    ap.add_argument("--max_eval", type=int, default=24)
    # --- optional visuals ---
    ap.add_argument("--visuals", action="store_true", help="also build the full floor + ply + walkthrough")
    ap.add_argument("--optimize", action="store_true"); ap.add_argument("--opt_iters", type=int, default=2000)
    ap.add_argument("--opt_lr", type=float, default=0.01); ap.add_argument("--opt_size", type=int, default=512)
    ap.add_argument("--view_points", type=int, default=2_000_000)
    ap.add_argument("--walk_size", type=int, default=768); ap.add_argument("--walk_fps", type=int, default=20)
    ap.add_argument("--walk_frames", type=int, default=150); ap.add_argument("--walk_steps", type=int, default=18)
    main(ap.parse_args())

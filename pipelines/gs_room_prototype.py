"""
Step-0 3D reconstruction prototype (Paper 2, the thesis goal).

SCIENTIFIC GOAL (kill-criterion first, no new model):
  Given per-room panoramas + metric depth placed at GROUND-TRUTH poses, how much of a
  novel view -- in particular one looking THROUGH A DOORWAY toward a neighbour room --
  is UNOBSERVED (disoccluded)? That unobserved fraction is exactly what a Step-1
  view-diffusion / inpainting prior would have to hallucinate. If it is tiny, the naive
  "unproject depth + splat" baseline is already close and the interesting research is
  elsewhere; if it is large, the completion prior IS the paper.

This deliberately SKIPS pose estimation and connectivity: we use GT poses so the only
thing measured is the single-/few-view coverage limit of unprojected-depth Gaussians.

Two scene builders share one disocclusion core:
  (A) --pano P --depth D            single room, one pano at identity; a NOVEL camera is
                                    synthesised by translating/rotating off the capture
                                    point. Fast local smoke test (needs no GT poses).
  (B) --home DIR --floor F          ZInD: auto-pick two CONNECTED rooms (panos in different
                                    rooms sharing a door), GT-pose + merge them, and put the
                                    novel camera AT the shared doorway looking into room B.
                                    This is the real Step-0 (run where panos+depth+poses coexist).

Metric (splat-sparsity-controlled):
  A point splat always has gaps between points, so a raw hole count is unfair. We measure
  coverage on a COARSE grid (a cell is "covered" if any Gaussian center lands in it) from
  (a) an original camera -- the best coverage the point density can give -- and (b) the
  novel camera. The DELTA (input_cov - novel_cov) is the disocclusion introduced purely by
  moving the viewpoint: the honest signal.

Outputs (results/gs_prototype/<tag>/): input.png, novel.png, coverage overlays, merged.ply,
  and a printed summary line. Nothing is dumped per-pano; a handful of files only.

Examples
  # local smoke test on an immersight capture (pano + dap depth present locally):
  python -m pipelines.gs_room_prototype --pano ".../panorama_1273530.png" \
      --depth ".../dap_depth/depth_meters/panorama_1273530.npy" \
      --baseline 1.0 --yaw 25 --tag immersight_smoke

  # real Step-0 on the cluster (ZInD, panos+depth+poses all present):
  python -m pipelines.gs_room_prototype --home $ZIND_ROOT/0072 --floor floor_01 \
      --depth_sub pager_depth/depth_meters --tag zind_0072
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import cv2

from sparsepano import config
from sparsepano.gs import gsplat_init as gi


# --------------------------------------------------------------------------- io
def _load_depth_2d(path):
    """Load depth and squeeze any singleton dim -> (H, W). PaGeR saves (1,H,W)."""
    d = np.load(path).astype(np.float32)
    d = np.squeeze(d)
    if d.ndim == 3:                                    # (H,W,1) or (1,H,W) leftover
        d = d[..., 0] if d.shape[-1] == 1 else d[0]
    if d.ndim != 2:
        raise ValueError(f"depth {path} has shape {np.load(path).shape}")
    return d


def _load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------- gaussian assembly
def build_room_gaussians(rgb, depth, pose_c2w, stride=4, max_depth=12.0):
    """Init per-pano Gaussians (camera frame) then transform to WORLD by GT c2w."""
    g = gi.gaussian_init_from_pano(depth, rgb, stride=stride, max_depth=max_depth)
    R, t = pose_c2w[:3, :3], pose_c2w[:3, 3]
    g["xyz"] = (g["xyz"] @ R.T + t).astype(np.float32)  # cam -> world
    return g


def merge(gs):
    return {k: np.concatenate([g[k] for g in gs], axis=0) for k in ("xyz", "rgb", "opacity", "scale", "rot")}


def _w2c_args(cam_c2w):
    """render_equirect applies xyz@R.T + t; supply (R,t) that map world -> camera."""
    R, C = cam_c2w[:3, :3], cam_c2w[:3, 3]
    return R.T, (-R.T @ C)


def _lookat_c2w(eye, target, up=(0.0, 1.0, 0.0)):
    """y-up world look-at -> camera-to-world (columns = right, up, forward)."""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    f = target - eye; f /= (np.linalg.norm(f) + 1e-9)
    r = np.cross(f, up); r /= (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, f, eye
    return T


# --------------------------------------------------------- gsplat surfel rendering (GPU)
def gsplat_render(g, cam_c2w, basis, fov, size, device="cuda"):
    """Real surfel render (rgb + ALPHA) of world Gaussians from a perspective camera.
    Reuses gs_optim's tested convention: effective c2w rotation = R_cam @ basis, viewmat =
    inverse. The alpha channel is the true coverage/hole map a point splat cannot give."""
    import torch
    from gsplat import rasterization
    from sparsepano.gs import gs_optim
    raw = gs_optim._to_raw(g, device)
    means, quats, scales, opac, colors = gs_optim._activated(raw)
    Reff = (cam_c2w[:3, :3] @ basis).astype(np.float32)
    T = np.eye(4, dtype=np.float32); T[:3, :3] = Reff; T[:3, 3] = cam_c2w[:3, 3]
    vm = torch.tensor(np.linalg.inv(T), device=device).float()[None]
    K = torch.tensor(gs_optim._K(fov, size), device=device).float()[None]
    rgb, alpha, _ = rasterization(means, quats, scales, opac, colors, vm, K,
                                  width=size, height=size, render_mode="RGB")
    rgb = rgb[0].clamp(0, 1).detach().cpu().numpy()
    alpha = alpha[0, ..., 0].detach().cpu().numpy()
    return (rgb * 255).astype(np.uint8), alpha


# ------------------------------------------------------------------ disocclusion metric
# A raw "any point hit this cell?" test saturates: a point splat has no surface, so far
# walls leak into would-be holes and every direction inside a room is filled. The honest
# proxy is DENSITY -- a disoccluded region is one the input cameras sampled densely but the
# novel view can only back with few, stretched points. We histogram point centres into
# coarse cells and threshold the count (tau). This is a stand-in for the alpha/hole map a
# real gaussian rasteriser (gsplat, cluster) produces; it is directional-solid-angle fair
# because equirect cells subtend comparable angle away from the poles.
from sparsepano.geometry import geom


def cell_counts(g_world, H, W, cam_c2w, grid=8):
    """Points-per-coarse-cell in the equirect view from cam_c2w."""
    R, t = _w2c_args(cam_c2w)
    xyz = g_world["xyz"] @ R.T + t
    u, v, r = geom.project_to_pano(xyz, W, H)
    gh, gw = H // grid, W // grid
    ci = np.clip((v / grid).astype(int), 0, gh - 1)
    cj = np.clip((u / grid).astype(int), 0, gw - 1)
    cnt = np.zeros((gh, gw), np.int64)
    np.add.at(cnt, (ci, cj), 1)
    return cnt


def render_and_cover(g, H, W, cam_c2w, grid=8):
    R, t = _w2c_args(cam_c2w)
    img, mask = gi.render_equirect(g, H, W, R=R, t=t)
    return img, mask


# --------------------------------------------------------------------------- modes
def _pose_translate(base_c2w, dx=0.0, dy=0.0, dz=0.0, yaw_deg=0.0):
    out = base_c2w.copy()
    if yaw_deg:
        a = np.deg2rad(yaw_deg); c, s = np.cos(a), np.sin(a)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
        out[:3, :3] = base_c2w[:3, :3] @ Ry
    out[:3, 3] = base_c2w[:3, 3] + base_c2w[:3, :3] @ np.array([dx, dy, dz], float)
    return out


def mode_single(a):
    rgb = _load_rgb(a.pano)
    depth = _load_depth_2d(a.depth)
    if rgb.shape[:2] != depth.shape:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    H, W = depth.shape
    eye = np.eye(4)
    g = build_room_gaussians(rgb, depth, eye, stride=a.stride, max_depth=a.max_depth)
    # novel camera: step sideways (+x) by `baseline` metres and yaw toward the wall
    novel = _pose_translate(eye, dx=a.baseline, yaw_deg=a.yaw)
    extras = {"rgbs": [rgb], "poses": [eye], "look_target": g["xyz"].mean(0)}
    return g, H, W, eye, novel, [("room", g)], extras


def _connected_pair(scene, tol_m=0.25):
    """Return (panoA, panoB, door) for two panos in DIFFERENT rooms that annotate the SAME
    physical door. ZInD annotates a shared door on both room boundaries but their world
    midpoints differ by ~5-10 cm, so we match by proximity (not exact uid). Among valid
    pairs, prefer the door whose two owning cameras are closest to it (cleanest geometry)."""
    inst = [(p, d, np.array(d.endpoints_xy).mean(0)) for p in scene.panos for d in p.doors]
    cands = []
    for i in range(len(inst)):
        pa, da, ma = inst[i]
        for j in range(i + 1, len(inst)):
            pb, db, mb = inst[j]
            if pa.room_id == pb.room_id:
                continue
            if np.linalg.norm(ma - mb) <= tol_m:
                # use door A's own midpoint (ma) — the same one used downstream (convention
                # check, novel cam) — so the straddle gate is consistent with what we return.
                Ca, Cb, Dm = pa.pose_c2w[[0, 2], 3], pb.pose_c2w[[0, 2], 3], ma
                va, vb = Ca - Dm, Cb - Dm
                straddle = np.degrees(np.arccos(np.clip(
                    va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9), -1, 1)))
                if straddle < 90:                             # cameras same side -> bad/overlapping pair
                    continue
                score = np.linalg.norm(va) + np.linalg.norm(vb)
                cands.append((score, pa, pb, da))
    if not cands:
        return None
    _, pa, pb, da = min(cands, key=lambda c: c[0])
    return pa, pb, da


def _depth_path_for(pano, home, depth_sub):
    return Path(home) / depth_sub / f"{pano.id}.npy"


def mode_zind(a):
    from sparsepano.datasets import zind
    root = Path(a.home).parent
    home_id = Path(a.home).name
    ds = zind.ZindDataset(root=str(root))
    scene = ds.scene(f"{home_id}/{a.floor}")
    pair = _connected_pair(scene)
    if pair is None:
        raise SystemExit(f"no connected room pair with a shared door in {home_id}/{a.floor}")
    pa, pb, door = pair
    print(f"[zind] connected pair: {pa.id} (room {pa.room_id}) <-> {pb.id} (room {pb.room_id}) via door {door.uid}")

    # optional pose-error injection on room B: perturb its GT pose by a yaw error and a
    # horizontal translation error (the thesis' real bottleneck — estimated, not GT, poses).
    # Room A stays at GT so the perturbation is purely the *relative* pose error.
    pose_b = pb.pose_c2w.copy()
    if a.pose_noise_deg or a.pose_noise_m:
        rng = np.random.default_rng(a.noise_seed)
        dyaw = np.deg2rad(a.pose_noise_deg) * rng.standard_normal()
        c, s = np.cos(dyaw), np.sin(dyaw)
        Rn = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
        pose_b[:3, :3] = Rn @ pose_b[:3, :3]
        ang = rng.uniform(0, 2 * np.pi)
        pose_b[[0, 2], 3] += a.pose_noise_m * np.array([np.cos(ang), np.sin(ang)])
        print(f"[zind] pose noise on room B: yaw {np.degrees(dyaw):+.1f} deg, "
              f"trans {a.pose_noise_m:.2f} m")

    gs, named, rgbs, poses = [], [], [], []
    for p, pose in ((pa, pa.pose_c2w), (pb, pose_b)):
        dp = _depth_path_for(p, a.home, a.depth_sub)
        if not dp.exists():
            raise SystemExit(f"missing depth {dp} -- generate depth for this home first")
        rgb = _load_rgb(p.image_path)
        depth = _load_depth_2d(dp)
        if rgb.shape[:2] != depth.shape:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        g = build_room_gaussians(rgb, depth, pose, stride=a.stride, max_depth=a.max_depth)
        gs.append(g); named.append((p.room_id, g)); rgbs.append(rgb); poses.append(pose)
    H, W = depth.shape
    merged = merge(gs)

    # convention self-check (mode B was untestable locally). Uses the two EXACT camera
    # positions (not room B's centroid, which drifts for large/L-shaped rooms and gave false
    # alarms). In a correct assembly the shared door sits BETWEEN the two cameras, so:
    #   (a) from A, the door bearing ~ the bearing to camera B  (door is toward B), and
    #   (b) the angle camA-door-camB is large (cameras straddle the door).
    def _az(v):
        return np.degrees(np.arctan2(v[0], v[1]))
    def _adiff(a, b):
        return abs((a - b + 180) % 360 - 180)
    Ca = pa.pose_c2w[[0, 2], 3]
    Cb = pb.pose_c2w[[0, 2], 3]
    Dm = np.array(door.endpoints_xy).mean(0)
    az_AB = _az(Cb - Ca); az_Ad = _az(Dm - Ca)
    toward_B = _adiff(az_AB, az_Ad)                            # door in direction of B from A?
    va, vb = Ca - Dm, Cb - Dm
    straddle = np.degrees(np.arccos(np.clip(
        va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9), -1, 1)))
    ok = (toward_B < 60) and (straddle > 90)
    print(f"[zind] convention check: door-toward-B={toward_B:5.1f}deg  "
          f"straddle(camA-door-camB)={straddle:5.1f}deg  "
          f"{'OK' if ok else '!! SUSPECT (flip or bad pair — inspect merged.ply/input.png)'}")

    # novel camera. Default: stand AT the shared doorway (x,z = door mid). With --novel_frac f
    # (0=camera A, 1=camera B) place it at a fraction along the A->B baseline instead, so you
    # can SWEEP how disocclusion grows as the viewpoint moves off the capture points and into
    # the neighbour room (the door is the *best* vantage, hence a lower bound).
    cam = pa.pose_c2w.copy()
    if getattr(a, "novel_frac", None) is not None:
        C = Ca + float(a.novel_frac) * (Cb - Ca)
    else:
        C = Dm
    cam[0, 3], cam[2, 3] = C[0], C[1]
    extras = {"rgbs": rgbs, "poses": poses, "look_target": gs[1]["xyz"].mean(0)}  # look at room B
    return merged, H, W, pa.pose_c2w, cam, named, door, extras


def main(a):
    door = None
    if a.pano:
        g, H, W, input_cam, novel_cam, named, extras = mode_single(a)
        merged = g
    else:
        merged, H, W, input_cam, novel_cam, named, door, extras = mode_zind(a)

    out = config.RESULTS_ROOT / "gs_prototype" / a.tag
    out.mkdir(parents=True, exist_ok=True)

    if a.rasterizer == "gsplat":
        return _main_gsplat(a, merged, input_cam, novel_cam, extras, door, out)

    # downscale render resolution for speed (metric geometry unchanged)
    rH, rW = min(H, a.render_h), min(H, a.render_h) * 2
    img_in, _ = render_and_cover(merged, rH, rW, input_cam, a.grid)
    img_nv, _ = render_and_cover(merged, rH, rW, novel_cam, a.grid)

    # density-based disocclusion. tau = a fraction of the input view's typical (median
    # non-empty) cell count -> a cell is "well observed" if it holds >= tau points.
    cnt_in = cell_counts(merged, rH, rW, input_cam, a.grid)
    cnt_nv = cell_counts(merged, rH, rW, novel_cam, a.grid)
    typical = np.median(cnt_in[cnt_in > 0]) if (cnt_in > 0).any() else 1.0
    tau = max(1.0, a.tau_frac * float(typical))
    well_in = cnt_in >= tau                                    # densely-observed content
    poor_nv = cnt_nv < tau
    denom = int(well_in.sum())
    new_holes = int((well_in & poor_nv).sum())
    disocc = new_holes / max(denom, 1)

    cv2.imwrite(str(out / "input.png"), cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "novel.png"), cv2.cvtColor(img_nv, cv2.COLOR_RGB2BGR))
    _save_disocc_overlay(out / "disocc.png", well_in, poor_nv, a.grid)
    gi.write_point_ply(str(out / "merged.ply"), merged)

    print(f"\n==== Step-0 disocclusion — tag={a.tag} ====")
    print(f"  gaussians          : {len(merged['xyz']):,}")
    print(f"  render res         : {rH}x{rW}   coarse grid {a.grid}px  tau={tau:.0f} (typical {typical:.0f})")
    print(f"  well-observed cells (input) : {denom}")
    print(f"  DISOCCLUSION       : {disocc:.3f}  ({new_holes}/{denom} densely-seen cells under-sampled from novel view)")
    if door is not None:
        print(f"  novel cam @ doorway, looking into neighbour room")
    print(f"  wrote {out}/input.png novel.png disocc.png merged.ply")


def _interior_holes(alpha, thr):
    """True disocclusion holes = low-alpha pixels ENCLOSED by covered geometry (not the empty
    background outside the scene silhouette). Flood-fill background from the border."""
    covered = (alpha >= thr).astype(np.uint8)
    h, w = covered.shape
    ff = (1 - covered).astype(np.uint8)                       # empty pixels
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 0)                        # kill empty region connected to border
    interior = covered | (ff > 0)                             # covered + enclosed empties
    holes = (alpha < thr) & (interior > 0)
    return interior.astype(bool), holes


def _main_gsplat(a, merged, input_cam, novel_cam, extras, door, out):
    """Real surfel render (gsplat, GPU). The alpha channel gives a TRUE hole map, which the
    CPU point splat cannot — this checks whether 'coverage is solved' survives proper rendering."""
    from sparsepano.gs import gs_optim
    thr, fov, size = a.alpha_thr, a.gs_fov, a.gs_size
    basis, vflip, psnr = gs_optim.auto_convention(merged, extras["rgbs"], extras["poses"],
                                                  fov=fov, size=min(size, 256))
    tgt = extras["look_target"]
    in_cam = _lookat_c2w(extras["poses"][0][:3, 3], tgt)      # from camera A, facing room B
    nv_cam = _lookat_c2w(novel_cam[:3, 3], tgt)               # from novel position, facing room B
    rgb_in, al_in = gsplat_render(merged, in_cam, basis, fov, size)
    rgb_nv, al_nv = gsplat_render(merged, nv_cam, basis, fov, size)

    int_in, hole_in = _interior_holes(al_in, thr)
    int_nv, hole_nv = _interior_holes(al_nv, thr)
    disocc_nv = hole_nv.sum() / max(int_nv.sum(), 1)
    disocc_in = hole_in.sum() / max(int_in.sum(), 1)          # control: ~0 if surfels fill

    cv2.imwrite(str(out / "gs_input.png"), cv2.cvtColor(rgb_in, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "gs_novel.png"), cv2.cvtColor(rgb_nv, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "gs_novel_alpha.png"), (np.clip(al_nv, 0, 1) * 255).astype(np.uint8))
    ov = cv2.cvtColor(rgb_nv, cv2.COLOR_RGB2BGR); ov[hole_nv] = (0, 0, 255)
    cv2.imwrite(str(out / "gs_novel_holes.png"), ov)

    print(f"\n==== Step-0 gsplat surfel — tag={a.tag} ====")
    print(f"  gaussians          : {len(merged['xyz']):,}   perspective {size}px fov{fov}")
    print(f"  convention (auto)  : basis-matched PSNR {psnr:.1f} dB  {'OK' if psnr > 18 else '!! LOW (convention/bleed)'}")
    print(f"  input-view holes (control) : {disocc_in:.3f}  (should be ~0 if surfels fill)")
    print(f"  NOVEL through-door DISOCCLUSION (true alpha holes): {disocc_nv:.3f}")
    print(f"  wrote {out}/gs_novel.png gs_novel_alpha.png gs_novel_holes.png gs_input.png")


def _save_disocc_overlay(path, well_in, poor_nv, grid):
    gh, gw = well_in.shape
    ov = np.zeros((gh, gw, 3), np.uint8)
    ov[well_in] = (80, 80, 80)                                 # observed content: grey
    ov[well_in & poor_nv] = (0, 0, 255)                        # disoccluded: red
    ov = cv2.resize(ov, (gw * grid, gh * grid), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), ov)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # mode A
    ap.add_argument("--pano", help="single-room mode: pano image path")
    ap.add_argument("--depth", help="single-room mode: depth .npy path")
    ap.add_argument("--baseline", type=float, default=1.0, help="novel-cam sideways step (m)")
    ap.add_argument("--yaw", type=float, default=20.0, help="novel-cam yaw (deg)")
    # mode B
    ap.add_argument("--home", help="ZInD mode: home dir (…/full_dataset/0072)")
    ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--depth_sub", default="dap_depth/depth_meters")
    ap.add_argument("--novel_frac", type=float, default=None,
                    help="place novel cam at this fraction along camA->camB (0=A,1=B); "
                         "default = at the door midpoint. Sweep to get the disocclusion curve.")
    ap.add_argument("--pose_noise_deg", type=float, default=0.0,
                    help="inject yaw error (std, deg) into room B's pose — the pose-sensitivity test")
    ap.add_argument("--pose_noise_m", type=float, default=0.0,
                    help="inject horizontal translation error (m) into room B's pose")
    ap.add_argument("--noise_seed", type=int, default=0)
    # rendering backend
    ap.add_argument("--rasterizer", choices=["points", "gsplat"], default="points",
                    help="'points' = CPU density proxy (default); 'gsplat' = GPU surfels + true alpha holes")
    ap.add_argument("--alpha_thr", type=float, default=0.5, help="gsplat: alpha below this = hole")
    ap.add_argument("--gs_fov", type=float, default=90.0, help="gsplat perspective FOV (deg)")
    ap.add_argument("--gs_size", type=int, default=512, help="gsplat render size (px, square)")
    # shared
    ap.add_argument("--stride", type=int, default=4, help="pano pixel stride for GS init")
    ap.add_argument("--max_depth", type=float, default=12.0)
    ap.add_argument("--render_h", type=int, default=512, help="render pano height (width=2h)")
    ap.add_argument("--grid", type=int, default=8, help="coarse coverage cell size (px)")
    ap.add_argument("--tau_frac", type=float, default=0.25,
                    help="cell is well-observed if it holds >= tau_frac * median-cell points")
    ap.add_argument("--tag", default="proto")
    a = ap.parse_args()
    if not a.pano and not a.home:
        ap.error("give either --pano/--depth (single) or --home/--floor (ZInD)")
    main(a)

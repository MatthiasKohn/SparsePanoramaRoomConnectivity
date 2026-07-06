"""
COLMAP via PERSPECTIVE SPLITTING — fix for the rejected "SPHERE" camera model.

Instead of fighting the build's missing spherical model, split each pano into
n overlapping PINHOLE ring views with KNOWN intrinsics + known ring yaw, run
standard COLMAP on the tiles, then collapse tile poses back to per-PANO poses:

  - all tiles of one pano share the camera center (pure-rotation rig)
    -> pano position = mean of its registered tile centers
       (center spread is reported: it should be ~cm, a rig sanity check);
  - tile at ring yaw psi views pano azimuth psi
    -> pano heading = circular mean over tiles of (tile forward azimuth - psi);
  - world up = mean of tile up axes (COLMAP is y-down); the world is rotated so
    up = +Y before azimuths/positions are read (gravity-aligned output).

Output: <home>/colmap_persp/pano_poses.json  {pano_stem: 4x4 cam-to-world}
consumed by exp24/exp27 via load_colmap (which accepts .json since this change).

  python scripts/colmap_perspective.py --home ../data/zind/full_dataset/0025 \
      --n_views 12 --fov 90 --size 1024            # full pipeline (needs colmap on PATH)
  python scripts/colmap_perspective.py --home ... --stage split      # tiles only
  python scripts/colmap_perspective.py --home ... --stage recover    # after colmap ran

If the recovered layout in exp24 looks MIRRORED vs GT (huge error, correct shape),
rerun recover with --flip_x (azimuth handedness fallback; similarity alignment
cannot absorb a reflection, and the project's empirically-calibrated ZInD
convention may be the mirror of COLMAP's physical world).
"""
import sys, os, argparse, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import cv2

from src import panoproj


# ---------------- stage 1: split ----------------
def split(home, n_views, fov, size, equi_w=4096):
    panos = sorted((Path(home) / "panos").glob("*.jpg"))
    if not panos:
        raise SystemExit(f"no panos at {home}/panos")
    work = Path(home) / "colmap_persp"
    tdir = work / "tiles"; tdir.mkdir(parents=True, exist_ok=True)
    meta = {}
    yaws = np.linspace(0, 360, n_views, endpoint=False)
    for p in panos:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (equi_w, equi_w // 2))
        for yaw in yaws:
            name = f"{p.stem}__y{int(round(yaw)):03d}.jpg"
            tile = panoproj.e2p(im, yaw, 0.0, fov, (size, size))
            cv2.imwrite(str(tdir / name), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
            meta[name] = dict(stem=p.stem, yaw=float(yaw))
    f = 0.5 * size / np.tan(np.radians(fov) / 2)
    json.dump(dict(f=f, cx=size / 2, cy=size / 2, size=size, fov=fov,
                   n_views=n_views, tiles=meta), open(work / "tiles_meta.json", "w"))
    print(f"[split] {len(panos)} panos -> {len(meta)} tiles at {tdir} (f={f:.1f}px)")
    return work


# ---------------- stage 2: colmap ----------------
def run_colmap(work, use_gpu=1, matcher="exhaustive"):
    meta = json.load(open(work / "tiles_meta.json"))
    db = work / "database.db"; sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    cam = f"{meta['f']},{meta['f']},{meta['cx']},{meta['cy']}"
    def run(*args):
        print("[colmap]", " ".join(args))
        subprocess.run(["colmap", *args], check=True)
    run("feature_extractor", "--database_path", str(db),
        "--image_path", str(work / "tiles"),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", cam,
        "--SiftExtraction.use_gpu", str(use_gpu))
    run(f"{matcher}_matcher", "--database_path", str(db),
        "--SiftMatching.use_gpu", str(use_gpu))
    run("mapper", "--database_path", str(db),
        "--image_path", str(work / "tiles"),
        "--output_path", str(sparse),
        "--Mapper.ba_refine_focal_length", "0",       # intrinsics are KNOWN
        "--Mapper.ba_refine_extra_params", "0")
    models = sorted(sparse.glob("[0-9]*"))
    if not models:
        print("[colmap] mapper produced NO model — the SALVe-style negative result: "
              "SfM cannot bridge these views. Record it; it is the motivation number.")
    return models


# ---------------- stage 3: recover pano poses ----------------
def _largest_model(sparse):
    import pycolmap
    best, bn = None, -1
    for d in sorted(Path(sparse).glob("[0-9]*")):
        rec = pycolmap.Reconstruction(str(d))
        if len(rec.images) > bn:
            best, bn = rec, len(rec.images)
    return best


def recover_poses(meta, tiles, flip_x=False, min_tiles=2):
    """Pure math (testable without pycolmap): tile cam-to-world 4x4s -> pano poses.
    meta: {tile_name: {stem, yaw}};  tiles: {tile_name: T_cam2world (OpenCV, y-down)}."""
    # gravity: world up = mean of tile up axes (-y col of cam-to-world R, y-down cam)
    ups = np.stack([-np.asarray(T)[:3, 1] for T in tiles.values()])
    up = ups.mean(0); up /= np.linalg.norm(up)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, y); c = float(up @ y)
    if np.linalg.norm(v) < 1e-8:
        Rup = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rup = np.eye(3) + K + K @ K / (1 + c)
    tilt = np.degrees(np.arccos(np.clip(ups @ up, -1, 1)))
    print(f"[recover] camera tilt spread vs global up: median {np.median(tilt):.2f} deg "
          f"(should be small — gravity-alignment check)")

    sx = -1.0 if flip_x else 1.0
    by_pano = {}
    for name, T in tiles.items():
        by_pano.setdefault(meta[name]["stem"], []).append((meta[name]["yaw"], np.asarray(T)))

    poses, report = {}, []
    for stem, lst in by_pano.items():
        if len(lst) < min_tiles:
            continue
        C = np.stack([Rup @ T[:3, 3] for _, T in lst])
        center = C.mean(0)
        spread = float(np.linalg.norm(C - center, axis=1).mean())
        angs = []
        for yaw, T in lst:
            fwd = Rup @ T[:3, 2]                       # OpenCV forward = +z
            az = np.arctan2(sx * fwd[0], fwd[2])
            angs.append(az - np.radians(yaw))
        th = float(np.arctan2(np.mean(np.sin(angs)), np.mean(np.cos(angs))))
        cth, sth = np.cos(th), np.sin(th)
        T = np.eye(4)
        T[:3, :3] = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
        T[0, 3], T[1, 3], T[2, 3] = sx * center[0], center[1], center[2]
        poses[stem] = T.tolist()
        report.append((stem, len(lst), spread))
    return poses, report


def recover(work, flip_x=False, min_tiles=2):
    meta = json.load(open(work / "tiles_meta.json"))["tiles"]
    rec = _largest_model(work / "sparse")
    if rec is None:
        raise SystemExit("no sparse model found — run stage colmap first")
    tiles = {}
    for img in rec.images.values():
        if img.name not in meta:
            continue
        T = np.eye(4); T[:3, :4] = img.cam_from_world.matrix()   # world->cam
        tiles[img.name] = np.linalg.inv(T)                        # cam->world
    print(f"[recover] {len(tiles)}/{len(meta)} tiles registered "
          f"in the largest model ({len(rec.images)} images)")
    poses, report = recover_poses(meta, tiles, flip_x=flip_x, min_tiles=min_tiles)

    out = work / "pano_poses.json"
    json.dump(poses, open(out, "w"))
    print(f"[recover] {len(poses)} pano poses -> {out}")
    for stem, n, sp in sorted(report):
        flag = "  <-- center spread high, treat with suspicion" if sp > 0.15 else ""
        print(f"    {stem}: {n} tiles, center spread {sp:.3f} (model units){flag}")
    home = work.parent
    print("\nnext:")
    print(f"  python experiments/exp24_colmap_compare.py --home {home} --model {out}")
    print(f"  python experiments/exp27_hybrid_real.py --home {home} "
          f"--depth_dir {home}/dap_depth/depth_meters --ckpt runs/hardneg/best.pt --model {out}")
    return poses


def main(a):
    work = Path(a.home) / "colmap_persp"
    if a.stage in ("all", "split"):
        work = split(a.home, a.n_views, a.fov, a.size)
    if a.stage in ("all", "colmap"):
        run_colmap(work, use_gpu=a.use_gpu, matcher=a.matcher)
    if a.stage in ("all", "recover"):
        recover(work, flip_x=a.flip_x, min_tiles=a.min_tiles)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--stage", choices=["all", "split", "colmap", "recover"], default="all")
    ap.add_argument("--n_views", type=int, default=12)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    ap.add_argument("--use_gpu", type=int, default=1)
    ap.add_argument("--flip_x", action="store_true",
                    help="negate x if exp24 shows a mirrored layout")
    ap.add_argument("--min_tiles", type=int, default=2)
    a = ap.parse_args()
    main(a)

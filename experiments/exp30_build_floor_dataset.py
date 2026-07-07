"""
exp30 — Build the FLOOR-LEVEL door dataset (the PaperV2 substrate).

Every v2 head (M1 correspondence / M2 distance+side / M3 floor attention) consumes the
same thing: per floor, ALL doors as tokens with GT supervision. This extractor produces,
per home/floor, a JSON with:
  panos[pano]     = { pose_se2:[x,z,yaw] (metric floor frame), cam_h_m }
  doors[i]        = { pano, bearing_deg, width_m, gt_dist_m (camera->door),
                      uid (shared physical-door id), global_xy, crop? }
  correspondences = [[i,j], ...]  door i,j (different panos) that are the SAME physical
                    door (uid match) -> the OT/Sinkhorn supervision.
  singletons      = doors whose uid appears once (to outside / unimaged room) -> the
                    dustbin cases the OT head must absorb.

GT geometry only (no depth, no detector, no GPU) -> validate anywhere. Crops (--with_crops)
reuse the e2p door crop so tokens can later be embedded by the trained encoder.

  python experiments/exp30_build_floor_dataset.py --home ../data/zind/full_dataset/0025 --floor floor_01
  python experiments/exp30_build_floor_dataset.py --root ../data/zind/full_dataset --max 50 --with_crops
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from itertools import combinations
import numpy as np

import config
from src import zind, door_dataset, panoproj


def floor_poses_se2(fl):
    """metric SE(2) pose (x, z, yaw) per pano, in the ZInD floor-plan frame."""
    S = fl.meters_per_coord
    out = {}
    for pano, info in fl.panos.items():
        x, z = info["pos"][0] * S, info["pos"][1] * S
        out[pano] = dict(pose_se2=[float(x), float(z), float(np.deg2rad(info["rot_deg"]))],
                         cam_h_m=float(info["cam_h_m"]))
    return out


def extract_floor(home_dir, floor, out_dir, fov=70.0, crop=(224, 224), tol=0.15, with_crops=False):
    jp = Path(home_dir) / "zind_data.json"
    fl = zind.ZindFloor(jp, floor=floor)            # raises on null scale -> skip upstream
    S = fl.meters_per_coord
    poses = floor_poses_se2(fl)

    # gather every door of every pano
    doors = []
    for pano, info in fl.panos.items():
        cam = np.array(info["pos"], float)
        for (d0, d1) in info["doors_global"]:
            mid = (d0 + d1) / 2.0
            doors.append(dict(pano=pano,
                              bearing_deg=float(door_dataset.door_azimuth(fl, pano, mid)),
                              width_m=float(np.linalg.norm(d0 - d1) * S),
                              gt_dist_m=float(np.linalg.norm(mid - cam) * S),
                              global_xy=[float(mid[0]), float(mid[1])]))

    # assign uid: doors within tol (coord units) are the SAME physical door
    uid = [-1] * len(doors); nxt = 0
    for i in range(len(doors)):
        if uid[i] != -1:
            continue
        uid[i] = nxt
        gi = np.array(doors[i]["global_xy"])
        for j in range(i + 1, len(doors)):
            if uid[j] == -1 and np.linalg.norm(np.array(doors[j]["global_xy"]) - gi) < tol:
                uid[j] = nxt
        nxt += 1
    for i, u in enumerate(uid):
        doors[i]["uid"] = int(u)

    # correspondences (same uid, different pano) and singletons (dustbin cases)
    from collections import defaultdict
    by_uid = defaultdict(list)
    for i, d in enumerate(doors):
        by_uid[d["uid"]].append(i)
    corr = []
    for u, idxs in by_uid.items():
        for a, b in combinations(idxs, 2):
            if doors[a]["pano"] != doors[b]["pano"]:
                corr.append([a, b])
    singletons = [idxs[0] for u, idxs in by_uid.items() if len(idxs) == 1]

    if with_crops:
        import cv2
        cdir = out_dir / "crops"; cdir.mkdir(parents=True, exist_ok=True)
        cache = {}
        pdir = Path(home_dir) / "panos"
        for i, d in enumerate(doors):
            if d["pano"] not in cache:
                im = cv2.imread(str(pdir / f"{d['pano']}.jpg"))
                cache[d["pano"]] = cv2.cvtColor(cv2.resize(im, (4096, 2048)), cv2.COLOR_BGR2RGB) if im is not None else None
            im = cache[d["pano"]]
            if im is None:
                d["crop"] = None; continue
            c = panoproj.e2p(im, d["bearing_deg"], 0, fov, crop)
            name = f"{Path(home_dir).name}_{floor}_{i:03d}.jpg"
            cv2.imwrite(str(cdir / name), cv2.cvtColor(c, cv2.COLOR_RGB2BGR)); d["crop"] = name

    rec = dict(home=Path(home_dir).name, floor=floor, meters_per_coord=float(S),
               panos=poses, doors=doors, correspondences=corr, singletons=singletons)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{Path(home_dir).name}_{floor}.json", "w") as f:
        json.dump(rec, f)
    return rec


def main(a):
    out = Path(a.out)
    homes = [Path(a.home)] if a.home else \
        sorted({p.parent for p in Path(a.root).glob("*/zind_data.json")})[:a.max]
    n_ok = n_doors = n_corr = n_single = 0
    for h in homes:
        for floor in ([a.floor] if a.floor else _floors(h)):
            try:
                rec = extract_floor(h, floor, out, a.fov, with_crops=a.with_crops)
            except Exception as e:
                continue
            n_ok += 1; n_doors += len(rec["doors"]); n_corr += len(rec["correspondences"])
            n_single += len(rec["singletons"])
            if a.home:
                print(f"{rec['home']} {floor}: {len(rec['panos'])} panos, {len(rec['doors'])} doors, "
                      f"{len(set(d['uid'] for d in rec['doors']))} unique doors, "
                      f"{len(rec['correspondences'])} correspondences, {len(rec['singletons'])} singletons")
                ds = np.array([d['gt_dist_m'] for d in rec['doors']])
                print(f"  camera->door distance: median {np.median(ds):.2f} m  range {ds.min():.2f}-{ds.max():.2f}")
    print(f"[done] {n_ok} floors -> {out}  ({n_doors} doors, {n_corr} correspondences, "
          f"{n_single} singletons/dustbin)")


def _floors(home_dir):
    try:
        return list(json.load(open(Path(home_dir) / "zind_data.json"))["merger"].keys())
    except Exception:
        return ["floor_01"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home"); ap.add_argument("--root")
    ap.add_argument("--floor", default=None)
    ap.add_argument("--out", default="data_floors")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--max", type=int, default=9999)
    ap.add_argument("--with_crops", action="store_true")
    a = ap.parse_args()
    main(a)

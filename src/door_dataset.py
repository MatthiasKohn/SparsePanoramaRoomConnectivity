"""
Build cross-view door-pair training data from ZInD.

A door shared by two adjacent rooms is seen from BOTH rooms' panoramas. We crop a
perspective view centred on that door in each pano -> a POSITIVE pair (same physical
door, opposite sides). Different doors are negatives (handled by the contrastive
batch). Labels are free from the ZInD floor plan -> no manual annotation.

Output: <out>/crops/<scene>_<doorid>_a.jpg / _b.jpg  + pairs.csv manifest.
"""
import csv
from itertools import combinations
from pathlib import Path
import numpy as np
import cv2

from src import zind, panoproj
from src.providers import ZIND_CONV


def _Tworld(info, S, conv=ZIND_CONV):
    cx, cy = info["pos"]
    if conv["swap"]:
        cx, cy = cy, cx
    yaw = np.deg2rad(conv["yaw_sign"] * info["rot_deg"] + conv["yaw_off"])
    T = np.eye(4); T[:3, :3] = zind_Ry(yaw)
    T[:3, 3] = [conv["sx"] * cx * S, info["cam_h_m"], conv["sz"] * cy * S]
    return T


def zind_Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def door_azimuth(fl, pano, global_mid, conv=ZIND_CONV):
    info = fl.panos[pano]; S = fl.meters_per_coord
    cx, cy = global_mid
    if conv["swap"]:
        cx, cy = cy, cx
    world = np.array([conv["sx"] * cx * S, info["cam_h_m"], conv["sz"] * cy * S])
    Tinv = np.linalg.inv(_Tworld(info, S, conv))
    loc = Tinv[:3, :3] @ world + Tinv[:3, 3]
    return float(np.degrees(np.arctan2(loc[0], loc[2])))


def extract_floor(json_path, panos_dir, out_dir, scene_id="000",
                  fov_deg=70.0, crop=(224, 224), tol=0.15, floor="floor_01"):
    fl = zind.ZindFloor(json_path, floor=floor)
    panos_dir, out_dir = Path(panos_dir), Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    rows = []
    cache = {}

    def rgb(stem):
        if stem not in cache:
            p = panos_dir / f"{stem}.jpg"
            im = cv2.imread(str(p))
            cache[stem] = cv2.cvtColor(cv2.resize(im, (4096, 2048)), cv2.COLOR_BGR2RGB) if im is not None else None
        return cache[stem]

    names = list(fl.panos)
    k = 0
    for a, b in combinations(names, 2):
        if fl.same_room(a, b):
            continue
        mid = fl.shared_door(a, b, tol=tol)
        if mid is None:
            continue
        ra, rb = rgb(a), rgb(b)
        if ra is None or rb is None:
            continue
        aza, azb = door_azimuth(fl, a, mid), door_azimuth(fl, b, mid)
        ca = panoproj.e2p(ra, aza, 0, fov_deg, crop)
        cb = panoproj.e2p(rb, azb, 0, fov_deg, crop)
        did = f"{scene_id}_{k:03d}"
        pa = out_dir / "crops" / f"{did}_a.jpg"
        pb = out_dir / "crops" / f"{did}_b.jpg"
        cv2.imwrite(str(pa), cv2.cvtColor(ca, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(pb), cv2.cvtColor(cb, cv2.COLOR_RGB2BGR))
        rows.append(dict(door_id=did, scene=scene_id, pano_a=a, pano_b=b,
                         az_a=round(aza, 1), az_b=round(azb, 1),
                         crop_a=pa.name, crop_b=pb.name))
        k += 1
    man = out_dir / "pairs.csv"
    write_header = not man.exists()
    with open(man, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                            ["door_id", "scene", "pano_a", "pano_b", "az_a", "az_b", "crop_a", "crop_b"])
        if write_header:
            wr.writeheader()
        wr.writerows(rows)
    return rows

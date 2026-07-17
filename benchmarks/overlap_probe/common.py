"""
overlap_probe.common — build evaluation SCENES from ZInD and hold ground truth.

The scientific question: do feed-forward panoramic reconstructors (Argus, PanoVGGT, ...)
actually recover camera poses when the capture is ONE PANO PER ROOM (near-zero overlap
except through doorways), or only when panos overlap? We answer it by running each model on
the SAME homes under two regimes and comparing to ZInD GT poses:

  regime "dense"  = every pano on the floor            (high overlap; where FMs should win)
  regime "sparse" = one pano per room (room-central)   (our target regime; near-zero overlap)

ZInD gives gravity-aligned metric GT: per pano a 2D floor position (meters) + yaw. We expose
GT as c2w 4x4 matrices (rotation = yaw about the vertical axis, translation = [x, 0, z] m).
"""
import os, json
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from sparsepano.datasets import zind


def _Rz_about_vertical(yaw_deg: float) -> np.ndarray:
    """Rotation by yaw about the +y (vertical/gravity) axis. Floor plane = x-z."""
    a = np.radians(yaw_deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], float)


@dataclass
class Scene:
    home: str
    floor: str
    regime: str                      # "dense" | "sparse"
    stems: list                      # pano stems, defines the ordering of everything below
    image_paths: list                # absolute paths, same order as stems
    gt_c2w: np.ndarray               # (N,4,4) ground-truth camera-to-world (metric, meters)
    rooms: list                      # room id per pano (same order)
    meters_per_coord: float
    overlap: dict = field(default_factory=dict)  # (i,j)->category, filled by overlap.py

    @property
    def n(self):
        return len(self.stems)


def _floors(home: Path):
    jp = home / "zind_data.json"
    if not jp.exists():
        return []
    try:
        d = json.load(open(jp))
    except Exception:
        return []
    return list(d.get("merger", {}).keys())


def _gt_c2w(fl: zind.ZindFloor, stem: str) -> np.ndarray:
    info = fl.panos[stem]
    x_m, z_m = np.asarray(info["pos"], float) * fl.meters_per_coord
    T = np.eye(4)
    T[:3, :3] = _Rz_about_vertical(info["rot_deg"])
    T[:3, 3] = [x_m, 0.0, z_m]
    return T


def _room_central_pano(fl: zind.ZindFloor, stems):
    """One representative pano per room: the one nearest its room's centroid."""
    by_room = {}
    for s in stems:
        by_room.setdefault(fl.panos[s]["room"], []).append(s)
    picked = []
    for room, ss in by_room.items():
        P = np.array([fl.panos[s]["pos"] for s in ss], float)
        c = P.mean(0)
        picked.append(ss[int(np.argmin(((P - c) ** 2).sum(1)))])
    return picked


def build_scenes(home_dir, min_rooms=3, min_panos=3):
    """Return {'dense': Scene|None, 'sparse': Scene|None, 'fl': ZindFloor, 'floor': str}."""
    home = Path(home_dir)
    panos_dir = home / "panos"
    out = {}
    for floor in _floors(home):
        try:
            fl = zind.ZindFloor(home / "zind_data.json", floor=floor)
        except Exception:
            continue
        stems = [s for s in fl.panos if (panos_dir / f"{s}.jpg").exists()]
        if len(stems) < min_panos:
            continue
        n_rooms = len({fl.panos[s]["room"] for s in stems})
        if n_rooms < min_rooms:
            continue

        def _mk(sel, regime):
            sel = sorted(sel)
            return Scene(
                home=home.name, floor=floor, regime=regime, stems=sel,
                image_paths=[str(panos_dir / f"{s}.jpg") for s in sel],
                gt_c2w=np.stack([_gt_c2w(fl, s) for s in sel]),
                rooms=[fl.panos[s]["room"] for s in sel],
                meters_per_coord=fl.meters_per_coord,
            )

        dense = _mk(stems, "dense")
        sparse_sel = _room_central_pano(fl, stems)
        sparse = _mk(sparse_sel, "sparse") if len(sparse_sel) >= min_rooms else None
        return {"dense": dense, "sparse": sparse, "fl": fl, "floor": floor}
    return {"dense": None, "sparse": None, "fl": None, "floor": None}


def iter_homes(root, only=None, limit=None):
    root = Path(root)
    homes = sorted({p.parent for p in root.glob("*/zind_data.json")})
    if only:
        keep = set(Path(only).read_text().split()) if Path(only).exists() \
            else set(str(only).split(","))
        homes = [h for h in homes if h.name in keep]
    if limit:
        homes = homes[:limit]
    return homes

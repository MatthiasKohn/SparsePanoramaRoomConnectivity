"""
ZInD floor-plan parser.

Reads zind_data.json and exposes, per panorama:
  - global 2D pose from floor_plan_transformation (translation, rotation deg, scale)
  - room id (for connectivity by shared room/door)
  - doors/windows/openings in BOTH local and global coords

Metric: floor-plan coordinate units -> meters via scale_meters_per_coordinate.

The mapping from ZInD's floor-plan frame to our equirectangular camera frame
(geom.py: X=right, Y=up, Z=forward, yaw about Y) is a fixed but unknown
sign/offset convention; it is calibrated empirically (see exp03 / calibrate).
"""
import json
from pathlib import Path
import numpy as np


def _rot2d(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _door_segments(flat):
    """ZInD W/D/O are stored as triplets; first two points are the endpoints."""
    segs = []
    arr = np.array(flat, dtype=float)
    for i in range(0, len(arr) - 1, 3):
        segs.append((arr[i], arr[i + 1]))
    return segs


class ZindFloor:
    def __init__(self, json_path, floor="floor_01"):
        d = json.load(open(json_path))
        self.meters_per_coord = d["scale_meters_per_coordinate"][floor]
        if self.meters_per_coord is None:
            raise ValueError(f"floor {floor!r} has null scale_meters_per_coordinate")
        self.panos = {}                       # stem -> info
        merger = d["merger"][floor]
        for room_key, room in merger.items():       # complete_room_XX
            for proom_key, proom in room.items():   # partial_room_XX
                for pano_key, pano in proom.items():  # pano_YY
                    stem = Path(pano["image_path"]).stem
                    tfm = pano["floor_plan_transformation"]
                    if tfm.get("scale") is None or tfm.get("translation") is None \
                       or tfm.get("rotation") is None:
                        continue            # incomplete transform -> skip this pano
                    R = _rot2d(tfm["rotation"]); sc = tfm["scale"]
                    tr = np.array(tfm["translation"], float)
                    lr = pano.get("layout_raw", {})
                    doors_local = _door_segments(lr.get("doors", []))
                    doors_global = [(sc * (R @ p0) + tr, sc * (R @ p1) + tr)
                                    for p0, p1 in doors_local]
                    self.panos[stem] = {
                        "room": room_key,
                        "pos": tr,                  # camera centre, coord units
                        "rot_deg": tfm["rotation"],
                        "scale": sc,
                        "cam_h_m": sc * self.meters_per_coord,  # height in meters
                        "doors_local": doors_local,
                        "doors_global": doors_global,
                        "label": pano.get("label", ""),
                    }

    def names(self):
        return list(self.panos)

    def same_room(self, a, b):
        return self.panos[a]["room"] == self.panos[b]["room"]

    def shared_door(self, a, b, tol=0.15):
        """Return (mid_global) of a door shared between a and b (in coord units),
        else None. Two rooms connect if a door of A coincides with a door of B."""
        for d0a, d1a in self.panos[a]["doors_global"]:
            ma = (d0a + d1a) / 2
            for d0b, d1b in self.panos[b]["doors_global"]:
                mb = (d0b + d1b) / 2
                if np.linalg.norm(ma - mb) < tol:
                    return (ma + mb) / 2
        return None

    def bearing_to(self, a, target_global):
        """Azimuth (rad) of a global point as seen from pano a, BEFORE the
        ZInD->camera convention is applied (i.e. in floor-plan frame, relative
        to the camera heading)."""
        p = (target_global - self.panos[a]["pos"])
        Rinv = _rot2d(-self.panos[a]["rot_deg"])
        loc = Rinv @ p
        return np.arctan2(loc[0], loc[1]), loc

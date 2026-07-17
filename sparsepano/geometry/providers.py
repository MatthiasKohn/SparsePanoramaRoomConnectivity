"""
Flexible pair providers so experiments don't care where pairs / GT come from.

  - StanfordProvider : Stanford 2D-3D-S, proxy connectivity (centre distance),
                       full GT pose + DAP depth. Fully validated.
  - ZindProvider     : ZInD, TRUE connectivity (labelled pair table), GT pose
                       from floor_plan_transformation under the empirically
                       calibrated ZInD->geometry convention (see ZIND_CONV).
"""
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from sparsepano import config
from sparsepano.geometry import geom
from sparsepano.datasets import stanford
from sparsepano.datasets import zind

# ZInD floor-plan -> our camera frame (geom.py), calibrated in exp03:
#   world X = -pos_x * S,  world Z = +pos_y * S,  yaw = +rotation_deg
# (best of 64 sign/offset/axis combos: 1.7x more tightly-consistent points
#  than median convention.)
ZIND_CONV = dict(sx=-1.0, sz=+1.0, swap=False, yaw_sign=+1.0, yaw_off=0.0)


@dataclass
class PairRecord:
    a: str
    b: str
    connected: bool
    dist: float


class StanfordProvider:
    name = "stanford"

    def __init__(self, area="area_3", max_dist=2.5, unconnected_min=5.0):
        self.names, self.P = stanford.list_panos(area)
        self.poses = {n: stanford.load_pose(n, self.P) for n in self.names}
        self.max_dist = max_dist
        self.unconnected_min = unconnected_min
        self._hw = {}

    def pairs(self, max_connected=12, max_unconnected=12, seed=0):
        rng = np.random.default_rng(seed)
        conn, unconn = [], []
        from itertools import combinations
        for na, nb in combinations(self.poses, 2):
            pa, pb = self.poses[na], self.poses[nb]
            if pa["room"] == pb["room"]:
                continue
            d = float(np.linalg.norm(pa["loc"] - pb["loc"]))
            if d < self.max_dist:
                conn.append(PairRecord(na, nb, True, d))
            elif d > self.unconnected_min:
                unconn.append(PairRecord(na, nb, False, d))
        conn.sort(key=lambda r: r.dist)
        rng.shuffle(unconn)
        return conn[:max_connected] + unconn[:max_unconnected]

    def hw(self, name):
        if name not in self._hw:
            self._hw[name] = stanford.get_hw(name, self.P)
        return self._hw[name]

    def depth(self, name):
        return stanford.load_dap_depth(name, self.P, self.hw(name))

    def rel_pose(self, a, b):
        return geom.rel_pose(self.poses[a], self.poses[b])

    def label(self, name):
        return self.poses[name]["room"]


class ZindProvider:
    name = "zind"

    def __init__(self, floor_json, pano_dir, depth_dir, pair_csv, conv=ZIND_CONV):
        self.fl = zind.ZindFloor(floor_json)
        self.pano_dir = Path(pano_dir)
        self.depth_dir = Path(depth_dir)
        self.conv = conv
        self.rows = list(csv.DictReader(open(pair_csv)))

    @staticmethod
    def _stem(p):
        return Path(p.replace("\\", "/")).stem

    def _has(self, name):
        return name in self.fl.panos and (self.depth_dir / f"{name}.npy").exists()

    def pairs(self, max_connected=12, max_unconnected=12, seed=0):
        out_c, out_u = [], []
        for r in self.rows:
            a, b = self._stem(r["pano_a"]), self._stem(r["pano_b"])
            if not (self._has(a) and self._has(b)):
                continue
            rec = PairRecord(a, b, r["connected"].lower() == "true", float(r["dist_m"]))
            (out_c if rec.connected else out_u).append(rec)
        return out_c[:max_connected] + out_u[:max_unconnected]

    def hw(self, name):
        return np.load(self.depth_dir / f"{name}.npy").shape

    def depth(self, name):
        return np.load(self.depth_dir / f"{name}.npy").astype(np.float32)

    def _Tworld(self, name):
        info = self.fl.panos[name]; S = self.fl.meters_per_coord
        cx, cy = info["pos"]
        if self.conv["swap"]:
            cx, cy = cy, cx
        X = self.conv["sx"] * cx * S
        Z = self.conv["sz"] * cy * S
        yaw = np.deg2rad(self.conv["yaw_sign"] * info["rot_deg"] + self.conv["yaw_off"])
        T = np.eye(4); T[:3, :3] = geom.Ry(yaw)
        T[:3, 3] = [X, info["cam_h_m"], Z]
        return T

    def rel_pose(self, a, b):
        return np.linalg.inv(self._Tworld(b)) @ self._Tworld(a)

    def shared_door_bearing(self, a, b):
        """Azimuth (rad) in A's pano of the door shared with B (None if not found).
        Uses the SAME convention so it lines up with backprojected points."""
        mid = self.fl.shared_door(a, b)
        if mid is None:
            return None
        # express the shared-door global point in A's camera frame, then azimuth
        Ta = self._Tworld(a)
        info = self.fl.panos[a]; S = self.fl.meters_per_coord
        cx, cy = mid
        if self.conv["swap"]:
            cx, cy = cy, cx
        world = np.array([self.conv["sx"] * cx * S, info["cam_h_m"], self.conv["sz"] * cy * S])
        loc = np.linalg.inv(Ta)[:3, :3] @ world + np.linalg.inv(Ta)[:3, 3]
        return np.arctan2(loc[0], loc[2])

    def label(self, name):
        return self.fl.panos[name]["label"] or self.fl.panos[name]["room"]


def default_zind():
    zp = config.zind_paths()
    legacy = config.PROJECT_ROOT.parent / "SemanticRoomConnection"
    pair_csv = legacy / "data" / "sample_pairs.csv"
    clean = zp["dap_meters"]
    legacy_depth = legacy / "data" / "depths" / "sample_tour" / "000" / "panos"
    depth_dir = next((c for c in (clean, legacy_depth)
                      if c.exists() and any(c.glob("*.npy"))), None)
    if depth_dir is None or not pair_csv.exists() or not zp["data_json"].exists():
        return None
    return ZindProvider(zp["data_json"], zp["panos"], depth_dir, pair_csv)

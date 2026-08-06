"""Structured3D scene adapter with the :class:`ZindFloor` interface.

Structured3D uses millimetres, +Z up, and panorama cameras whose centre column
looks along global -Y.  The rest of this repository expects horizontal
coordinates ``(world X, world Z)`` and uses a reflected camera-to-world pose for
ZInD.  We therefore store Structured3D ``(x, y)`` as ``(-x, y)`` and use a
constant ``rot_deg=180``.  In this frame ``pose_c2w_gt`` has the documented
Structured3D forward direction and panorama handedness.  Junctions, camera
centres, doors, and room polygons all receive the same transform.

The adapter intentionally has no Shapely dependency.  Annotation IDs are
resolved through their explicit ``ID`` fields, while incidence matrices use
the list indices prescribed by the official Structured3D tools.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_CONFIGS = {"empty", "simple", "full"}
_OPENING_TYPES = {"window", "opening"}


def _rot2d(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=float)


def _xy(points: np.ndarray) -> np.ndarray:
    """Official (x,y,z) mm -> repository horizontal (-x,y) mm."""
    points = np.asarray(points, dtype=float)
    return np.stack([-points[..., 0], points[..., 1]], axis=-1)


def _cycles(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Join unordered incidence-matrix edges into closed polygon cycles."""
    unused = [tuple(map(int, edge)) for edge in edges if len(set(edge)) == 2]
    out: list[list[int]] = []
    while unused:
        a, b = unused.pop(0)
        chain = [a, b]
        while chain[-1] != chain[0]:
            end = chain[-1]
            hit = next((i for i, edge in enumerate(unused) if end in edge), None)
            if hit is None:
                break
            u, v = unused.pop(hit)
            chain.append(v if u == end else u)
        if len(chain) >= 4 and chain[-1] == chain[0]:
            out.append(chain[:-1])
    return out


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (a[1] > y) != (b[1] > y):
            xhit = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < xhit:
                inside = not inside
    return inside


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    t = np.clip(np.dot(point - a, d) / max(np.dot(d, d), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * d)))


def _boundary_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    return min(_point_segment_distance(point, a, b)
               for a, b in zip(polygon, np.roll(polygon, -1, axis=0)))


class Structured3DFloor:
    """Load one ``scene_<id>`` directory as a floor.

    Args:
        scene_dir: Directory containing ``annotation_3d.json`` and
            ``2D_rendering``.  Passing the dataset root is deliberately not
            supported because ``--home`` denotes one floor in the pipeline.
        config: Furniture configuration: ``empty``, ``simple``, or ``full``.

    ``rgb_path(stem)`` and each info dictionary's ``image_path`` provide the
    stable equirectangular RGB location.  All required ZindFloor fields are
    populated, including normalized camera/ceiling heights used by the layout
    depth renderer.
    """

    meters_per_coord = 0.001

    def __init__(self, scene_dir: str | Path, config: str = "full"):
        self.scene_dir = Path(scene_dir)
        if config not in _CONFIGS:
            raise ValueError(f"config must be one of {sorted(_CONFIGS)}, got {config!r}")
        self.config = config
        annotation_path = self.scene_dir / "annotation_3d.json"
        rendering_dir = self.scene_dir / "2D_rendering"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Structured3D annotation not found: {annotation_path}")
        if not rendering_dir.is_dir():
            raise FileNotFoundError(f"Structured3D panoramas not found: {rendering_dir}")

        with annotation_path.open(encoding="utf8") as f:
            anno = json.load(f)
        junctions = np.asarray([j["coordinate"] for j in anno["junctions"]], dtype=float)
        plane_line = np.asarray(anno["planeLineMatrix"])
        line_junction = np.asarray(anno["lineJunctionMatrix"])
        planes = anno["planes"]
        plane_index = {int(p.get("ID", i)): i for i, p in enumerate(planes)}

        def plane_cycles(plane_id: int) -> list[list[int]]:
            pi = plane_index[int(plane_id)]
            line_ids = np.flatnonzero(plane_line[pi]).tolist()
            edges = []
            for li in line_ids:
                js = np.flatnonzero(line_junction[li]).tolist()
                if len(js) == 2:
                    edges.append((js[0], js[1]))
            return _cycles(edges)

        # A room semantic owns a floor plane.  Its explicit semantic ID is the
        # GT room ID; camera containment links rendering folder IDs robustly.
        rooms: list[dict] = []
        apertures: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = []
        for sem in anno["semantics"]:
            sem_type = str(sem["type"])
            sem_planes = [int(pid) for pid in sem.get("planeID", [])]
            floor_ids = [pid for pid in sem_planes
                         if planes[plane_index[pid]]["type"] == "floor"]
            if floor_ids and sem_type not in {"outwall", "door", "window", "opening"}:
                cycles = plane_cycles(floor_ids[0])
                if cycles:
                    cycle = max(cycles, key=len)
                    floor_z = float(np.median(junctions[cycle, 2]))
                    ceil_zs: list[float] = []
                    for pid in sem_planes:
                        if planes[plane_index[pid]]["type"] == "ceiling":
                            for cyc in plane_cycles(pid):
                                ceil_zs.extend(junctions[cyc, 2].tolist())
                    rooms.append({
                        "id": str(sem.get("ID", len(rooms))),
                        "label": sem_type,
                        "polygon": _xy(junctions[cycle]),
                        "floor_z": floor_z,
                        "ceil_z": float(np.median(ceil_zs)) if ceil_zs else None,
                    })
            if sem_type == "door" or sem_type in _OPENING_TYPES:
                for pid in sem_planes:
                    js = sorted({j for cyc in plane_cycles(pid) for j in cyc})
                    if len(js) < 2:
                        continue
                    pts = _xy(junctions[js])
                    # A vertical rectangular aperture has two unique projected
                    # endpoints.  Farthest-pair also handles redundant vertices.
                    best = max(((np.linalg.norm(pts[i] - pts[j]), pts[i], pts[j])
                                for i in range(len(pts)) for j in range(i + 1, len(pts))),
                               key=lambda item: item[0])
                    if best[0] > 1e-6:
                        apertures.append((sem_type, (best[1].copy(), best[2].copy())))

        if not rooms:
            raise ValueError(f"no room floor semantics found in {annotation_path}")

        self.panos: dict[str, dict] = {}
        for room_dir in sorted(p for p in rendering_dir.iterdir() if p.is_dir()):
            pano_dir = room_dir / "panorama"
            image_path = pano_dir / config / "rgb_rawlight.png"
            camera_path = pano_dir / "camera_xyz.txt"
            if not image_path.is_file() or not camera_path.is_file():
                continue
            camera = np.asarray(np.loadtxt(camera_path), dtype=float).reshape(-1)
            if camera.size < 3:
                raise ValueError(f"expected x y z in {camera_path}")
            pos = _xy(camera[:3])
            containing = [r for r in rooms if _point_in_polygon(pos, r["polygon"])]
            room = min(containing or rooms,
                       key=lambda r: _boundary_distance(pos, r["polygon"]))
            cam_h_mm = float(camera[2] - room["floor_z"])
            if cam_h_mm <= 0:
                raise ValueError(f"camera is not above its room floor: {camera_path}")
            ceil_z = room["ceil_z"] if room["ceil_z"] is not None else camera[2] + cam_h_mm
            rot_deg = 180.0
            Rinv = _rot2d(-rot_deg)
            verts_global = room["polygon"].copy()
            verts_local = (verts_global - pos) @ Rinv.T
            self.panos[room_dir.name] = {
                "room": room["id"],
                "pos": pos,
                "rot_deg": rot_deg,
                "scale": 1.0,
                "cam_h_m": cam_h_mm * self.meters_per_coord,
                "doors_local": [],
                "doors_global": [],
                "openings_global": [],
                "verts_local": verts_local,
                "verts_global": verts_global,
                "camera_height": 1.0,
                "ceiling_height": float((ceil_z - room["floor_z"]) / cam_h_mm),
                "label": room["label"],
                "image_path": image_path,
            }

        if not self.panos:
            raise FileNotFoundError(
                f"no '{config}' rgb_rawlight panoramas with camera_xyz.txt under {rendering_dir}")

        # Attach the exact same global segment to every incident room.  This is
        # what makes a physical door resolve identically from both sides.
        info_by_room = {info["room"]: info for info in self.panos.values()}
        for kind, segment in apertures:
            mid = (segment[0] + segment[1]) / 2.0
            distances = sorted((_boundary_distance(mid, room["polygon"]), room["id"])
                               for room in rooms if room["id"] in info_by_room)
            incident = [rid for dist, rid in distances if dist <= 100.0]
            # Annotation rounding can put an aperture slightly off a wall.  A
            # second room is accepted only within 25 cm; exterior doors remain
            # attached to one room rather than creating false adjacency.
            if len(incident) < 2:
                incident = [rid for dist, rid in distances[:2] if dist <= 250.0]
            for rid in incident:
                info = info_by_room[rid]
                key = "doors_global" if kind == "door" else "openings_global"
                info[key].append((segment[0].copy(), segment[1].copy()))
                if kind == "door":
                    Rinv = _rot2d(-info["rot_deg"])
                    info["doors_local"].append(
                        (Rinv @ (segment[0] - info["pos"]),
                         Rinv @ (segment[1] - info["pos"])))

    def names(self) -> list[str]:
        return list(self.panos)

    def rgb_path(self, stem: str) -> Path:
        return Path(self.panos[stem]["image_path"])

    def same_room(self, a: str, b: str) -> bool:
        return self.panos[a]["room"] == self.panos[b]["room"]

    def shared_door(self, a: str, b: str, tol: float = 0.15):
        """Return a shared door midpoint in stored (millimetre) coordinates."""
        for d0a, d1a in self.panos[a]["doors_global"]:
            ma = (d0a + d1a) / 2.0
            for d0b, d1b in self.panos[b]["doors_global"]:
                mb = (d0b + d1b) / 2.0
                if np.linalg.norm(ma - mb) < tol:
                    return (ma + mb) / 2.0
        return None

    def bearing_to(self, a: str, target_global):
        """Return panorama azimuth (radians) and local horizontal vector."""
        p = np.asarray(target_global, dtype=float) - self.panos[a]["pos"]
        loc = _rot2d(-self.panos[a]["rot_deg"]) @ p
        return np.arctan2(loc[0], loc[1]), loc

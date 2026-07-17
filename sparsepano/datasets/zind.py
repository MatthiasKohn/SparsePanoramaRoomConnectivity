"""ZInD dataset adapter.

ZInD-specific JSON structure and scale conventions stay in this file. Pipelines
and metrics should consume the generic ``Scene``/``Pano``/``Door`` dataclasses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from sparsepano.datasets import zind_floor as legacy_zind

from .base import Dataset, Door, Pano, Scene
from .registry import register_dataset
from .zind_floor import ZindFloor


def _pose_c2w(info: dict, meters_per_unit: float) -> np.ndarray:
    yaw = np.deg2rad(float(info["rot_deg"]))
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )
    pos = np.asarray(info["pos"], dtype=float) * meters_per_unit
    pose[:3, 3] = [pos[0], float(info.get("cam_h_m") or 0.0), pos[1]]
    return pose


def _door_uid(mid_xy_m: np.ndarray, tol_m: float = 0.05) -> str:
    # Stable enough to link the same physical door across adjacent rooms while
    # avoiding dependence on ZInD's nested room/pano field names.
    q = np.round(mid_xy_m / tol_m).astype(int)
    return f"door_{q[0]}_{q[1]}"


@register_dataset("zind")
class ZindDataset(Dataset):
    """Adapter for a ZInD ``full_dataset`` root."""

    def __init__(
        self,
        root: str,
        split_file: str | None = None,
        heldout_file: str | None = None,
    ):
        self.root = Path(root)
        self.split_file = Path(split_file) if split_file else None
        self.heldout_file = Path(heldout_file) if heldout_file else None

    def splits(self) -> dict[str, list[str]]:
        scene_ids = self._all_scene_ids()
        out = {"all": scene_ids}
        if self.heldout_file and self.heldout_file.exists():
            homes = set(self.heldout_file.read_text().split())
            out["heldout"] = [sid for sid in scene_ids if sid.split("/")[0] in homes]
        if self.split_file and self.split_file.exists():
            data = json.load(open(self.split_file))
            for name, ids in data.items():
                wanted = set(ids)
                out[name] = [sid for sid in scene_ids if sid in wanted or sid.split("/")[0] in wanted]
        return out

    def scenes(self, split: str | None = None) -> Iterable[Scene]:
        ids = self.splits().get(split or "all")
        if ids is None:
            raise KeyError(f"unknown split {split!r}; available: {sorted(self.splits())}")
        for scene_id in ids:
            yield self.scene(scene_id)

    def scene(self, scene_id: str) -> Scene:
        home_id, floor = self._parse_scene_id(scene_id)
        home = self.root / home_id
        fl = legacy_zind.ZindFloor(home / "zind_data.json", floor=floor)
        meters = float(fl.meters_per_coord)
        panos: list[Pano] = []
        for pano_id, info in sorted(fl.panos.items()):
            doors = []
            for p0, p1 in info["doors_global"]:
                p0m = np.asarray(p0, dtype=float) * meters
                p1m = np.asarray(p1, dtype=float) * meters
                mid = (p0m + p1m) / 2.0
                bearing = float(fl.bearing_to(pano_id, (p0 + p1) / 2.0)[0])
                bearing_deg = float(np.degrees(bearing))
                width_m = float(np.linalg.norm(p1m - p0m))
                doors.append(
                    Door(
                        pano_id=pano_id,
                        bearing_deg=bearing_deg,
                        width_m=width_m,
                        endpoints_xy=((float(p0m[0]), float(p0m[1])), (float(p1m[0]), float(p1m[1]))),
                        uid=_door_uid(mid),
                    )
                )
            depth_path = home / "dap_depth" / "depth_meters" / f"{pano_id}.npy"
            panos.append(
                Pano(
                    id=pano_id,
                    image_path=str(home / "panos" / f"{pano_id}.jpg"),
                    room_id=str(info["room"]),
                    pose_c2w=_pose_c2w(info, meters),
                    cam_height_m=info.get("cam_h_m"),
                    gt_depth_path=str(depth_path) if depth_path.exists() else None,
                    doors=doors,
                )
            )
        return Scene(
            dataset="zind",
            scene_id=f"{home_id}/{floor}",
            panos=panos,
            meters_per_unit=meters,
            caps={
                "gt_poses": True,
                "gt_depth": False,
                "gt_doors": True,
                "gt_rooms": True,
            },
        )

    def _all_scene_ids(self) -> list[str]:
        ids: list[str] = []
        for jp in sorted(self.root.glob("*/zind_data.json")):
            home_id = jp.parent.name
            try:
                data = json.load(open(jp))
            except Exception:
                continue
            for floor, scale in data.get("scale_meters_per_coordinate", {}).items():
                if scale is not None:
                    ids.append(f"{home_id}/{floor}")
        return ids

    @staticmethod
    def _parse_scene_id(scene_id: str) -> tuple[str, str]:
        if "/" in scene_id:
            home_id, floor = scene_id.split("/", 1)
            return home_id, floor
        return scene_id, "floor_01"


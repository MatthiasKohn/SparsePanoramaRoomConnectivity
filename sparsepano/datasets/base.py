"""Dataset-agnostic scene contract.

Everything outside ``sparsepano.datasets`` should consume these dataclasses
rather than dataset-specific JSON fields or folder conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Door:
    """A doorway/opening observed from one panorama."""

    pano_id: str
    bearing_deg: float
    width_m: float | None = None
    endpoints_xy: tuple[tuple[float, float], tuple[float, float]] | None = None
    uid: str | None = None


@dataclass(frozen=True)
class Pano:
    """One panorama and optional ground-truth annotations."""

    id: str
    image_path: str
    room_id: str
    pose_c2w: np.ndarray | None = None
    cam_height_m: float | None = None
    gt_depth_path: str | None = None
    doors: list[Door] = field(default_factory=list)


@dataclass(frozen=True)
class Scene:
    """One dataset scene, usually a home/floor."""

    dataset: str
    scene_id: str
    panos: list[Pano]
    meters_per_unit: float
    caps: dict[str, bool]


class Dataset(ABC):
    """Dataset adapter interface.

    A new dataset should implement this interface and register itself in
    ``sparsepano.datasets.registry``. Capability flags tell evaluators which
    metrics are scientifically valid.
    """

    name: str

    @abstractmethod
    def scenes(self, split: str | None = None) -> Iterable[Scene]:
        """Yield scenes, optionally restricted to a named split."""

    @abstractmethod
    def scene(self, scene_id: str) -> Scene:
        """Load a single scene by id."""

    @abstractmethod
    def splits(self) -> dict[str, list[str]]:
        """Return available split ids."""


"""Factory for the legacy floor-object interface consumed by pipelines."""

from pathlib import Path

from .structured3d_floor import Structured3DFloor
from .zind_floor import ZindFloor


def load_floor(dataset: str, home: str | Path, floor: str = "floor_01",
               config: str = "full"):
    """Return a ZindFloor-compatible floor without dataset checks downstream."""
    home = Path(home)
    if dataset == "zind":
        return ZindFloor(home / "zind_data.json", floor=floor)
    if dataset == "structured3d":
        return Structured3DFloor(home, config=config)
    raise ValueError(f"unknown floor dataset {dataset!r}")

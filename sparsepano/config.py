"""Project configuration and small config helpers.

This replaces the former root-level ``config.py``. Values can still be driven by
cluster environment variables such as ``PROJECT_ROOT`` and ``RESULTS_ROOT``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = PROJECT_ROOT.parent / "data"

# ---- Stanford 2D-3D-S ----
STANFORD_ROOT = DATA_ROOT / "standord2d3d"
STANFORD_GT_DEPTH_UNIT = 1.0 / 512.0
STANFORD_INVALID = 65535


def stanford_area(area: str = "area_3") -> dict:
    a = STANFORD_ROOT / area
    return {
        "root": a,
        "rgb": a / "pano" / "rgb",
        "pose": a / "pano" / "pose",
        "gt_depth": a / "pano" / "depth",
        "semantic": a / "pano" / "semantic",
        "dap_depth": a / "dap_depth" / "depth_meters",
    }


# ---- ZInD sample tour defaults ----
ZIND_TOUR = DATA_ROOT / "zind" / "sample_tour" / "000"


def zind_paths() -> dict:
    return {
        "panos": ZIND_TOUR / "panos",
        "dap_meters": ZIND_TOUR / "dap_depth" / "depth_meters",
        "data_json": ZIND_TOUR / "zind_data.json",
    }


# ---- DAP depth model repo lives beside this project ----
DAP_ROOT = PROJECT_ROOT.parent / "DAP"
DAP_RUNNER = DAP_ROOT / "test" / "run_dap_inference.py"

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", PROJECT_ROOT / "results")).resolve()
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    split_file: str | None = None
    heldout_file: str | None = None


@dataclass(frozen=True)
class RunConfig:
    dataset: DatasetConfig
    out: str = "results/dev"


def load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            f"{path} looks like YAML, but PyYAML is not installed. "
            "Install pyyaml or use JSON for this config."
        ) from exc
    return yaml.safe_load(text)


def expand_env(value: str | None) -> str | None:
    return os.path.expandvars(value) if value is not None else None


def dataset_config_from_mapping(data: dict[str, Any]) -> DatasetConfig:
    ds = data.get("dataset", data)
    return DatasetConfig(
        name=ds["name"],
        root=expand_env(ds["root"]) or ds["root"],
        split_file=expand_env(ds.get("split_file")),
        heldout_file=expand_env(ds.get("heldout_file")),
    )

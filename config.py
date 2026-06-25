"""
Central configuration. All dataset paths are resolved RELATIVE to the project
folder, so this works unchanged on the laptop and in any mounted sandbox as long
as `data/` sits next to `SparsePanoramaRoomConnectivity/`.

Layout assumed:
    Promotion/
      data/standord2d3d/area_3/...
      DAP/                              <- depth model repo
      SparsePanoramaRoomConnectivity/   <- this project
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent / "data"

# ---- Stanford 2D-3D-S ----
STANFORD_ROOT = DATA_ROOT / "standord2d3d"          # note: dataset folder name kept as-is
STANFORD_GT_DEPTH_UNIT = 1.0 / 512.0                # uint16 -> meters; 65535 == invalid
STANFORD_INVALID = 65535

def stanford_area(area: str = "area_3") -> dict:
    a = STANFORD_ROOT / area
    return {
        "root": a,
        "rgb":   a / "pano" / "rgb",
        "pose":  a / "pano" / "pose",
        "gt_depth": a / "pano" / "depth",
        "semantic": a / "pano" / "semantic",
        "dap_depth": a / "dap_depth" / "depth_meters",   # metric (verified scale ~1.0 vs GT)
    }

# ---- ZInD ----
ZIND_TOUR = DATA_ROOT / "zind" / "sample_tour" / "000"

def zind_paths() -> dict:
    return {
        "panos": ZIND_TOUR / "panos",
        "dap_meters": ZIND_TOUR / "dap_depth" / "depth_meters",
        "data_json": ZIND_TOUR / "zind_data.json",
    }

# ---- DAP (depth model) lives beside the project ----
DAP_ROOT = PROJECT_ROOT.parent / "DAP"
DAP_RUNNER = DAP_ROOT / "test" / "run_dap_inference.py"

RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_ROOT.mkdir(exist_ok=True)

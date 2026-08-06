import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from pipelines import floor as floor_pipeline
from sparsepano.datasets.floor_factory import load_floor
from sparsepano.datasets.structured3d_floor import Structured3DFloor
from sparsepano.geometry.layout_depth import render_layout_depth
from sparsepano.providers import connectivity, poses


def _write_scene(root: Path) -> Path:
    scene = root / "scene_00000"
    junctions = []
    lines = []
    planes = []
    plane_line = []
    line_junction = []

    def add_plane(points, plane_type):
        js = []
        for point in points:
            js.append(len(junctions))
            junctions.append({"ID": len(junctions), "coordinate": list(map(float, point))})
        edge_ids = []
        for a, b in zip(js, js[1:] + js[:1]):
            edge_ids.append(len(lines))
            lines.append({"ID": len(lines), "point": [0, 0, 0], "direction": [0, 0, 0]})
            row = [0] * len(junctions)
            row[a] = row[b] = 1
            line_junction.append(row)
        # Previously created rows need columns for this plane's new junctions.
        for row in line_junction[:-len(edge_ids)]:
            row.extend([0] * (len(junctions) - len(row)))
        pid = len(planes)
        planes.append({"ID": pid, "type": plane_type, "normal": [0, 0, 1], "offset": 0})
        plane_line.append(edge_ids)
        return pid

    room_specs = [
        (10, "living room", [(0, 0), (4000, 0), (4000, 4000), (0, 4000)]),
        (20, "bedroom", [(4000, 0), (8000, 0), (8000, 4000), (4000, 4000)]),
        (30, "kitchen", [(0, 4000), (4000, 4000), (4000, 8000), (0, 8000)]),
    ]
    semantics = []
    for rid, label, xy in room_specs:
        floor_id = add_plane([(x, y, 0) for x, y in xy], "floor")
        ceil_id = add_plane([(x, y, 3000) for x, y in xy], "ceiling")
        semantics.append({"ID": rid, "type": label, "planeID": [floor_id, ceil_id]})

    door_ab = add_plane([(4000, 1500, 0), (4000, 2500, 0),
                         (4000, 2500, 2100), (4000, 1500, 2100)], "wall")
    door_ac = add_plane([(1500, 4000, 0), (2500, 4000, 0),
                         (2500, 4000, 2100), (1500, 4000, 2100)], "wall")
    semantics.extend([
        {"ID": 100, "type": "door", "planeID": [door_ab]},
        {"ID": 101, "type": "door", "planeID": [door_ac]},
    ])

    # Convert sparse plane->line ID lists to the official binary matrix.
    nlines = len(lines)
    incidence = []
    for edge_ids in plane_line:
        row = [0] * nlines
        for edge_id in edge_ids:
            row[edge_id] = 1
        incidence.append(row)
    annotation = {
        "junctions": junctions,
        "lines": lines,
        "planes": planes,
        "semantics": semantics,
        "planeLineMatrix": incidence,
        "lineJunctionMatrix": line_junction,
        "cuboids": [],
        "manhattan": [],
    }
    scene.mkdir(parents=True)
    (scene / "annotation_3d.json").write_text(json.dumps(annotation))

    cameras = {"room_a": (2000, 2000, 1600), "room_b": (6000, 2000, 1600),
               "room_c": (2000, 6000, 1600)}
    for name, camera in cameras.items():
        pano = scene / "2D_rendering" / name / "panorama"
        pano.mkdir(parents=True)
        (pano / "camera_xyz.txt").write_text(" ".join(map(str, camera)))
        (pano / "layout.txt").write_text("")
        for cfg, color in (("empty", 32), ("simple", 96), ("full", 160)):
            cfg_dir = pano / cfg
            cfg_dir.mkdir()
            Image.fromarray(np.full((8, 16, 3), color, np.uint8)).save(cfg_dir / "rgb_rawlight.png")
    return scene


def test_structured3d_floor_contract_and_gt_self_consistency(tmp_path):
    scene = _write_scene(tmp_path)
    fl = Structured3DFloor(scene, config="full")
    assert fl.meters_per_coord == 0.001
    assert fl.names()
    required = {
        "pos", "rot_deg", "scale", "cam_h_m", "room", "doors_local",
        "doors_global", "openings_global", "verts_local", "verts_global",
        "camera_height", "ceiling_height", "label",
    }
    for stem, info in fl.panos.items():
        assert required <= info.keys()
        assert np.asarray(info["pos"]).shape == (2,)
        assert len(info["verts_global"]) >= 3
        assert fl.rgb_path(stem).is_file()

    assert fl.shared_door("room_a", "room_b") is not None
    assert fl.shared_door("room_a", "room_c") is not None
    assert fl.shared_door("room_b", "room_c") is None

    build, render, gt = poses.get_poses(fl, fl.meters_per_coord, model="gt")
    rmse, rot_error = floor_pipeline.pose_errors(gt, build, fl.names())
    rooms, adjacency = connectivity.get_rooms(fl, fl.names(), model="gt")
    door = floor_pipeline.door_consistency(
        fl, rooms, adjacency, build, gt, fl.meters_per_coord)
    assert rmse < 1e-10
    assert rot_error < 1e-10
    assert door["n_doors"] == 2
    assert door["door_gap_m"] == 0.0
    depth = render_layout_depth(fl, "room_a", H=32, W=64, mask_doors=False)
    assert depth.shape == (32, 64)
    assert np.count_nonzero(depth > 0) > depth.size // 2


def test_structured3d_configs_share_geometry(tmp_path):
    scene = _write_scene(tmp_path)
    floors = [load_floor("structured3d", scene, config=cfg)
              for cfg in ("empty", "simple", "full")]
    for stem in floors[0].names():
        reference = floors[0].panos[stem]
        for fl in floors[1:]:
            info = fl.panos[stem]
            np.testing.assert_array_equal(info["pos"], reference["pos"])
            np.testing.assert_array_equal(info["verts_global"], reference["verts_global"])
            assert len(info["doors_global"]) == len(reference["doors_global"])
            assert fl.rgb_path(stem) != floors[0].rgb_path(stem)


def test_structured3d_metrics_only_pipeline(tmp_path, monkeypatch):
    scene = _write_scene(tmp_path)
    results = tmp_path / "results"
    monkeypatch.setattr(floor_pipeline.config, "RESULTS_ROOT", results)
    args = Namespace(
        home=str(scene), dataset="structured3d", config="full", floor="floor_01", tag="smoke",
        metrics_only=True, pose_model="gt", noise_deg=0.0, noise_m=0.0, seed=0,
        pose_file="", depth_model="gt_layout", connectivity="gt", completion="none", gs_h=32,
    )
    floor_pipeline.main(args)
    with (results / "floor" / "results.csv").open(newline="") as f:
        row = list(csv.DictReader(f))[-1]
    assert float(row["pose_rmse_m"]) == 0.0
    assert float(row["door_gap_m"]) == 0.0
    assert int(row["n_doors"]) == 2

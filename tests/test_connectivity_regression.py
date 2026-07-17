"""Connectivity regression checks.

These tests are slow because they need the ZInD dataset and model weights. They
are intentionally opt-in for local/HPC verification:

    pytest tests/test_connectivity_regression.py --run-slow \
      --zind-root "$ZIND_ROOT" --heldout "$RUN_ROOT/hardneg/val_homes.txt" \
      --ckpt "$RUN_ROOT/hardneg/best.pt"
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("doors,target", [("gt", 0.913), ("detected", 0.842)])
def test_connectivity_ap_regression(pytestconfig, tmp_path, doors, target):
    if not pytestconfig.getoption("--run-slow"):
        pytest.skip("slow regression requires --run-slow and ZInD/model inputs")
    root = pytestconfig.getoption("--zind-root")
    heldout = pytestconfig.getoption("--heldout")
    ckpt = pytestconfig.getoption("--ckpt")
    if not root or not heldout:
        pytest.skip("missing --zind-root or --heldout")
    out = tmp_path / f"conn_{doors}"
    cmd = [
        sys.executable,
        "-m",
        "pipelines.run",
        "--dataset",
        "zind",
        "--root",
        root,
        "--only",
        heldout,
        "--split",
        "heldout",
        "--stage",
        "connectivity",
        "--doors",
        doors,
        "--ckpt",
        ckpt,
        "--scoring",
        "assign",
        "--max",
        "200",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    import json

    metrics = json.loads((out / "metrics.json").read_text())
    assert abs(metrics["ap_mean"] - target) < 0.04

"""
generate_depth.py — one-command DAP depth generation for this project.

Thin wrapper around YOUR existing runner (../DAP/test/run_dap_inference.py) so you
don't duplicate model-loading / scaling logic. It just points DAP at the right
input folder and writes depths where this project's providers expect them:
    <output>/depth_meters/<stem>.npy   (metric, scale 102.0)
    <output>/depth_npy/<stem>.npy      (raw)
    <output>/depth_vis/<stem>.png      (sanity colour map)

Run on the laptop (needs the DAP repo + weights + ideally a GPU):

    # Fill in ZInD depths for the whole sample tour (what exp03 needs):
    python scripts/generate_depth.py --dataset zind

    # Re-generate Stanford area_3 (already present, usually not needed):
    python scripts/generate_depth.py --dataset stanford

    # Any custom folder:
    python scripts/generate_depth.py --input_dir <imgs> --output_dir <out> --pattern "*.jpg"

This script is intentionally NOT runnable in the sandbox (no GPU / no DAP repo);
it is meant for your machine.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def _presets():
    sa = config.stanford_area("area_3")
    zp = config.zind_paths()
    return {
        "zind": dict(input_dir=zp["panos"],
                     output_dir=zp["dap_meters"].parent,   # .../dap_depth
                     pattern="*.jpg"),
        "stanford": dict(input_dir=sa["rgb"],
                         output_dir=sa["dap_depth"].parent,  # .../dap_depth
                         pattern="*.png"),
    }


def _load_dap_runner():
    if not config.DAP_RUNNER.exists():
        sys.exit(f"DAP runner not found at {config.DAP_RUNNER}\n"
                 f"Expected the DAP repo beside this project: {config.DAP_ROOT}")
    spec = importlib.util.spec_from_file_location("dap_runner", config.DAP_RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)         # safe: model load happens only on call
    return mod


def main():
    ap = argparse.ArgumentParser(description="DAP depth generation for this project")
    ap.add_argument("--dataset", choices=["zind", "stanford"], default=None)
    ap.add_argument("--input_dir", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--pattern", type=str, default=None)
    a = ap.parse_args()

    if a.dataset:
        p = _presets()[a.dataset]
        input_dir, output_dir, pattern = p["input_dir"], p["output_dir"], p["pattern"]
    elif a.input_dir and a.output_dir:
        input_dir = Path(a.input_dir); output_dir = Path(a.output_dir)
        pattern = a.pattern  or "*.png"
    else:
        ap.error("give --dataset OR (--input_dir and --output_dir)")

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}  (depth_meters/<stem>.npy)")
    print(f"Pattern: {pattern}")
    n = len(list(Path(input_dir).glob(pattern)))
    print(f"Images matched: {n}")
    if n == 0:
        sys.exit("No images matched — check --input_dir / --pattern.")

    dap = _load_dap_runner()
    dap.process_folder(Path(input_dir), Path(output_dir), pattern=pattern)

    meters = Path(output_dir) / "depth_meters"
    done = len(list(meters.glob("*.npy")))
    print(f"\nDone. {done} depth maps in {meters}")
    print("Providers will now pick these up (ZindProvider prefers this location).")


if __name__ == "__main__":
    main()

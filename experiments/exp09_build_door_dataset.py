"""
exp09 — Build the cross-view door-pair dataset from ZInD (all downloaded homes).

Scans a ZInD root for `<home>/zind_data.json` (+ `<home>/panos/`), extracts every
shared door from each floor as a positive pair of perspective crops, and appends to
one dataset dir (crops/ + pairs.csv). 'scene' column = home_floor (for scene-disjoint
train/val splits later).

  # one home (the sample tour) — quick check:
  python experiments/exp09_build_door_dataset.py --zind_root ../data/zind/sample_tour --out data_doorpairs
  # the full download (after running data/zind/download_data.py):
  python experiments/exp09_build_door_dataset.py --zind_root <ZIND_DOWNLOAD_DIR> --out data_doorpairs
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from src import door_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zind_root", required=True)
    ap.add_argument("--out", default="data_doorpairs")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--append", action="store_true",
                    help="add to an existing dataset instead of rebuilding it")
    ap.add_argument("--max_homes", type=int, default=None,
                    help="cap number of homes (for data-scaling experiments)")
    a = ap.parse_args()
    import shutil
    if Path(a.out).exists() and not a.append:
        print(f"clearing existing {a.out} (use --append to keep it)")
        shutil.rmtree(a.out)
    root = Path(a.zind_root)
    jsons = sorted(root.glob("**/zind_data.json"))
    if a.max_homes:
        jsons = jsons[:a.max_homes]
    print(f"found {len(jsons)} homes under {root}")
    total = 0
    for jp in jsons:
        home = jp.parent
        panos = home / "panos"
        if not panos.exists():
            continue
        floors = list(json.load(open(jp)).get("merger", {}).keys())
        for fl in floors:
            try:
                # ZindFloor defaults to floor_01; pass floor through door_dataset
                rows = door_dataset.extract_floor(jp, panos, a.out,
                                                  scene_id=f"{home.name}_{fl}",
                                                  fov_deg=a.fov, floor=fl)
                total += len(rows)
                print(f"  {home.name}/{fl}: +{len(rows)} pairs")
            except Exception as e:
                print(f"  [skip] {home.name}/{fl}: {e}")
    print(f"\nTOTAL cross-view door pairs: {total}  ->  {a.out}/pairs.csv")


if __name__ == "__main__":
    main()

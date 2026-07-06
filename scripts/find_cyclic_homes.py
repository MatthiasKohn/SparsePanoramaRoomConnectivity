"""
Scan ZInD homes and rank floors by shared-door graph cyclicity + size, to decide
WHERE to spend DAP depth generation (exp29 needs depth; cyclic floors are the
scientifically valuable ones — exp17: cycles are where geometry can compete, trees
are where the embedding prior is necessary).

Outputs scripts/depth_homes.txt (home ids, cyclic-first) + a CSV with per-floor stats.
CPU-only, fast (json parsing only).

  python scripts/find_cyclic_homes.py --root ../data/zind/full_dataset \
      --only runs/hardneg/val_homes.txt --top 20
"""
import sys, os, argparse, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from itertools import combinations

from src import zind


def floor_stats(jp, floor):
    fl = zind.ZindFloor(jp, floor=floor)
    rooms = sorted({info["room"] for info in fl.panos.values()})
    reps = {}
    for stem, info in fl.panos.items():
        reps.setdefault(info["room"], []).append(stem)
    edges = set()
    for ra, rb in combinations(rooms, 2):
        if any(fl.shared_door(x, y) is not None for x in reps[ra] for y in reps[rb]):
            edges.add((ra, rb))
    # largest connected component
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    seen, best = set(), set()
    for s0 in list(adj):
        if s0 in seen:
            continue
        st, comp = [s0], set()
        while st:
            u = st.pop()
            if u in comp:
                continue
            comp.add(u); st += list(adj[u])
        seen |= comp
        if len(comp) > len(best):
            best = comp
    e_cc = sum(1 for a, b in edges if a in best and b in best)
    cycles = e_cc - (len(best) - 1) if best else 0
    return dict(rooms=len(rooms), panos=len(fl.panos), edges=len(edges),
                cc_rooms=len(best), cycles=max(cycles, 0))


def main(a):
    homes = sorted({p.parent for p in Path(a.root).glob("**/zind_data.json")})
    if a.only:
        keep = set(Path(a.only).read_text().split())
        homes = [h for h in homes if h.name in keep]
        print(f"restricted to {len(homes)} homes from {a.only}")
    rows = []
    for h in homes[:a.max]:
        try:
            d = json.load(open(h / "zind_data.json"))
            for f, s in d["scale_meters_per_coordinate"].items():
                if s is None:
                    continue
                st = floor_stats(h / "zind_data.json", f)
                if st["cc_rooms"] < a.min_rooms:
                    continue
                st.update(home=h.name, floor=f)
                rows.append(st)
        except Exception as e:
            print(f"{h.name}: skipped ({e})")
    rows.sort(key=lambda r: (-r["cycles"], -r["cc_rooms"]))

    out_dir = Path(__file__).parent
    with open(out_dir / "cyclic_floors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["home", "floor", "rooms", "panos",
                                          "edges", "cc_rooms", "cycles"])
        w.writeheader(); w.writerows(rows)

    picked, seen = [], set()
    for r in rows:
        if r["home"] not in seen:
            seen.add(r["home"]); picked.append(r["home"])
        if len(picked) >= a.top:
            break
    (out_dir / "depth_homes.txt").write_text("\n".join(picked) + "\n")
    ncyc = sum(1 for r in rows if r["cycles"] > 0)
    print(f"{len(rows)} floors scanned, {ncyc} with >=1 cycle "
          f"({100*ncyc/max(len(rows),1):.0f}% — the C4 tree-dominance number)")
    print(f"top {len(picked)} homes -> {out_dir/'depth_homes.txt'}")
    for r in rows[:a.top]:
        print(f"  {r['home']}/{r['floor']}: {r['cc_rooms']} rooms, "
              f"{r['edges']} edges, {r['cycles']} cycles, {r['panos']} panos")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--only", help="restrict to held-out homes (val_homes.txt)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--max", type=int, default=9999)
    ap.add_argument("--min_rooms", type=int, default=4)
    a = ap.parse_args()
    main(a)

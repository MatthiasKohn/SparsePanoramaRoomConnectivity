"""
exp29 — Multi-floor pose/flip BENCHMARK (NextStage.md G2).

The flip/pose results so far rest on n=2 floors (sample tour 5/6, home 0025 7/7).
This batches the exp18/exp27 pipeline over every depth-equipped floor and aggregates
the paper's pose-side table:

  per floor : #rooms, #door edges, #cycles, mean inlier,
              flip accuracy {chance-geometry, free-space, embedding prior, solved graph},
              median layout error (m) {geometry-only, +embedding},
              stratified bridge-vs-cycle flip accuracy (the C4 claim).

Homes are given as directories that contain zind_data.json + panos/ and a depth dir
(default <home>/dap_depth/depth_meters, produced by scripts/generate_depth.py).

  python experiments/exp29_floor_benchmark.py --root ../data/zind/full_dataset \
      --homes scripts/depth_homes.txt --ckpt runs/hardneg/best.pt --device cuda
  python experiments/exp29_floor_benchmark.py --root ../data/zind/full_dataset \
      --homes 0025 --embed selftest          # CPU plumbing check on one home
"""
import sys, os, argparse, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src import posegraph as pg
from experiments.exp18_floor_graph_real import (measured_edge, side_pref_embedding,
                                                load_pano, gt2d, wall_profile,
                                                inside_fraction, get_embedder)
from experiments.exp27_hybrid_real import make_prov, door_edges, largest_cc


def floors_of(home_dir):
    d = json.load(open(Path(home_dir) / "zind_data.json"))
    return [f for f, s in d["scale_meters_per_coordinate"].items() if s is not None]


def bridge_mask(N, edges):
    """True for edges NOT on any cycle (removal disconnects the graph)."""
    out = []
    for k in range(len(edges)):
        adj = {}
        for kk, (i, j) in enumerate(edges):
            if kk == k:
                continue
            adj.setdefault(i, set()).add(j); adj.setdefault(j, set()).add(i)
        i0, j0 = edges[k]
        seen, st = set(), [i0]
        while st:
            u = st.pop()
            if u in seen:
                continue
            seen.add(u); st += list(adj.get(u, ()))
        out.append(j0 not in seen)
    return np.array(out)


def eval_floor(prov, embed, name, seed=0):
    de = door_edges(prov)
    if len(de) < 2:
        return None
    roomof = lambda s: prov.fl.panos[s]["room"]
    edge_rooms = [(roomof(x), roomof(y)) for x, y, _, _ in de]
    keep = largest_cc(edge_rooms)
    rep = {}
    for (x, y, _, _) in de:
        rep.setdefault(roomof(x), x); rep.setdefault(roomof(y), y)
    de = [t for t, (ra, rb) in zip(de, edge_rooms) if ra in keep and rb in keep]
    edge_rooms = [(ra, rb) for (ra, rb) in edge_rooms if ra in keep and rb in keep]
    stems = sorted(keep); idx = {r: k for k, r in enumerate(stems)}; N = len(stems)
    if N < 3 or len(de) < 2:
        return None
    Xgt = np.array([gt2d(prov, rep[r]) for r in stems])

    profs, pim = {}, {}
    edges, cand, weights, true_idx, fs_pref, emb_pref = [], [], [], [], [], []
    for (x, y, inl, e), (ra, rb) in zip(de, edge_rooms):
        i, j = idx[ra], idx[rb]
        m = e["m"]; mflip = pg.rot_pi_about(m, e["doorA"])
        edges.append((i, j)); cand.append((m, mflip)); weights.append(float(inl))
        mgt = pg.between(Xgt[i], Xgt[j])
        true_idx.append(0 if np.linalg.norm((m - mgt)[:2]) <=
                        np.linalg.norm((mflip - mgt)[:2]) else 1)
        if x not in profs:
            profs[x] = wall_profile(prov.depth(x))
        f0 = inside_fraction(profs[x], prov.depth(y), m)
        f1 = inside_fraction(profs[x], prov.depth(y), mflip)
        fs_pref.append(0 if f0 <= f1 else 1)
        if embed is not None:
            for s in (x, y):
                if s not in pim:
                    pim[s] = load_pano(prov, s)
            p = side_pref_embedding(embed, pim[x], pim[y], prov.depth(x),
                                    np.degrees(e["az_a"]), m, mflip)
            emb_pref.append(p)
    true_idx = np.array(true_idx); weights = np.array(weights)
    ncyc = len(edges) - (N - 1)
    br = bridge_mask(N, edges)

    sol_geo = pg.optimize(N, edges, cand, fixed0=Xgt[0], weights=weights,
                          restarts=10, seed=seed)
    res = dict(home=name, rooms=N, edges=len(edges), cycles=ncyc,
               bridges=int(br.sum()), mean_inlier=float(weights.mean()))

    def err(sol):
        A = pg.align_similarity(sol["X"][:, :2], Xgt[:, :2])
        return np.linalg.norm(A - Xgt[:, :2], axis=1)

    res["err_geo"] = float(np.median(err(sol_geo)))
    res["flip_geo"] = float((np.array(sol_geo["sel"]) == true_idx).mean())
    res["fs_prior_acc"] = float((np.array(fs_pref) == true_idx).mean())
    if embed is not None:
        ep = np.array(emb_pref); valid = ep != -1
        res["emb_prior_acc"] = float((ep[valid] == true_idx[valid]).mean()) if valid.any() else np.nan
        res["emb_valid"] = int(valid.sum())
        pref = [int(p) if p != -1 else -1 for p in ep]
        sol_emb = pg.optimize(N, edges, cand, fixed0=Xgt[0], weights=weights,
                              pref=pref, beta=0.5, restarts=10, seed=seed)
        sel = np.array(sol_emb["sel"])
        res["err_emb"] = float(np.median(err(sol_emb)))
        res["flip_emb"] = float((sel == true_idx).mean())
        # C4 stratification: bridge edges vs cycle edges
        if br.any():
            res["flip_emb_bridge"] = float((sel[br] == true_idx[br]).mean())
            res["flip_geo_bridge"] = float((np.array(sol_geo["sel"])[br] == true_idx[br]).mean())
        if (~br).any():
            res["flip_emb_cycle"] = float((sel[~br] == true_idx[~br]).mean())
            res["flip_geo_cycle"] = float((np.array(sol_geo["sel"])[~br] == true_idx[~br]).mean())
    return res


def main(a):
    root = Path(a.root)
    if Path(a.homes).exists():
        home_ids = Path(a.homes).read_text().split()
    else:
        home_ids = a.homes.split(",")

    embed = get_embedder(a.ckpt or a.embed, device=a.device) if (a.ckpt or a.embed) else None

    rows = []
    for hid in home_ids:
        home = root / hid
        depth_dir = Path(a.depth_sub) if os.path.isabs(a.depth_sub) else home / a.depth_sub
        if not depth_dir.exists():
            print(f"{hid}: no depth at {depth_dir} — skipped (run scripts/generate_depth.py)")
            continue
        for floor in floors_of(home):
            try:
                prov = make_prov(str(home), str(depth_dir), floor)
                have = [n for n in prov.fl.panos if (prov.depth_dir / f"{n}.npy").exists()]
                if len(have) < 3:
                    continue
                r = eval_floor(prov, embed, f"{hid}/{floor}", seed=a.seed)
            except Exception as e:
                print(f"{hid}/{floor}: FAILED — {e}")
                continue
            if r is None:
                continue
            rows.append(r)
            msg = (f"{r['home']:16} rooms {r['rooms']:2d} edges {r['edges']:2d} "
                   f"cyc {r['cycles']:2d} | geo: {r['err_geo']:.2f} m flip {r['flip_geo']:.2f}")
            if "flip_emb" in r:
                msg += f" | +emb: {r['err_emb']:.2f} m flip {r['flip_emb']:.2f} (prior {r['emb_prior_acc']:.2f})"
            print(msg)
    if not rows:
        print("no evaluable floors"); return

    def agg(key):
        v = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])])
        return (v.mean(), len(v)) if len(v) else (np.nan, 0)

    print("=" * 78)
    print(f"FLOORS: {len(rows)} | mean flip acc — geometry {agg('flip_geo')[0]:.2f}, "
          f"free-space prior {agg('fs_prior_acc')[0]:.2f}"
          + (f", EMBEDDING prior {agg('emb_prior_acc')[0]:.2f}, solved+emb {agg('flip_emb')[0]:.2f}"
             if embed is not None else ""))
    print(f"median layout err — geometry {agg('err_geo')[0]:.2f} m"
          + (f", +embedding {agg('err_emb')[0]:.2f} m" if embed is not None else ""))
    if embed is not None:
        print(f"C4 strata — bridge edges: geo {agg('flip_geo_bridge')[0]:.2f} vs "
              f"+emb {agg('flip_emb_bridge')[0]:.2f} | cycle edges: geo "
              f"{agg('flip_geo_cycle')[0]:.2f} vs +emb {agg('flip_emb_cycle')[0]:.2f}")

    out = config.RESULTS_ROOT / "floorbench"; out.mkdir(parents=True, exist_ok=True)
    tag = ("_" + a.tag) if a.tag else ""
    keys = sorted({k for r in rows for k in r})
    with open(out / f"floorbench{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    fg = [r["flip_geo"] for r in rows]
    ax[0].scatter([r["cycles"] for r in rows], fg, label="geometry", c="#999", s=30)
    if embed is not None:
        fe = [r.get("flip_emb", np.nan) for r in rows]
        ax[0].scatter([r["cycles"] for r in rows], fe, label="+embedding", c="#1f77b4", s=30)
    ax[0].axhline(0.5, color="r", ls="--", lw=1, label="chance")
    ax[0].set_xlabel("#cycles"); ax[0].set_ylabel("flip accuracy"); ax[0].legend(fontsize=8)
    ax[0].set_title("flip accuracy vs graph cyclicity", fontsize=10); ax[0].set_ylim(0, 1.05)
    eg = [r["err_geo"] for r in rows]
    ax[1].scatter(eg, [r.get("err_emb", np.nan) for r in rows], c="#1f77b4", s=30)
    lim = max(max(eg), 0.1) * 1.1
    ax[1].plot([0, lim], [0, lim], "k--", lw=1)
    ax[1].set_xlabel("geometry-only layout err (m)"); ax[1].set_ylabel("+embedding (m)")
    ax[1].set_title("layout error: below diagonal = embedding helps", fontsize=10)
    ax[2].hist([r.get("emb_prior_acc", np.nan) for r in rows], bins=10, range=(0, 1),
               color="#1f77b4", alpha=.85)
    ax[2].axvline(0.5, color="r", ls="--", lw=1)
    ax[2].set_title("embedding prior accuracy distribution", fontsize=10)
    fig.suptitle(f"exp29 floor benchmark over {len(rows)} floors", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out / f"floorbench{tag}.png"; fig.savefig(p, dpi=120); print("saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ZInD full_dataset root")
    ap.add_argument("--homes", required=True,
                    help="file with home ids (one per line) OR comma-separated ids")
    ap.add_argument("--depth_sub", default="dap_depth/depth_meters",
                    help="depth dir relative to each home (or absolute)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--embed", default=None, help="'selftest' for CPU plumbing")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a)

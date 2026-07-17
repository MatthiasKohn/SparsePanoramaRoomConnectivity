"""
exp17 — Complete FLOOR GRAPH from ambiguous door edges (semantic alignment).

Goal: given a set of panos on one floor, recover ONE globally consistent layout
(a complete room-connectivity graph with poses), resolving the per-edge
which-side flip that no single edge can resolve. This is the floor-level
foundation the 3D reconstruction needs.

Validates the BACK-END on ZInD GT:
  rooms = one pano per room;  edges = shared-door adjacency (GT);
  each edge -> TWO candidate relative poses {true, flip} (flip = point-reflection
  of B about the shared door = the which-side ambiguity); the TRUE one is hidden
  at a random index and both are noised. The SE(2) pose graph (sparsepano/pose/posegraph) must
  pick flips and place rooms by CYCLE CONSISTENCY (+ an optional appearance prior).

Key result: cycles resolve flips only on cyclic edges; BRIDGE edges (no cycle)
need an appearance prior -> what the door embedding (exp16) provides.

  python legacy/experiments/exp17_floor_graph.py --home 0025 --floor floor_01
  python legacy/experiments/exp17_floor_graph.py --home 0009 --p_app 0.85
"""
import sys, os, argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.datasets import zind
from sparsepano.pose import posegraph as pg

ZROOT = config.DATA_ROOT / "zind" / "full_dataset"


def build_floor(home, floor):
    fl = zind.ZindFloor(ZROOT / home / "zind_data.json", floor=floor)
    S = fl.meters_per_coord
    rep = {}
    for name, info in fl.panos.items():
        rep.setdefault(info["room"], name)
    rooms = list(rep)
    idx = {r: k for k, r in enumerate(rooms)}
    pose = {}
    for r in rooms:
        info = fl.panos[rep[r]]
        pose[r] = np.array([info["pos"][0] * S, info["pos"][1] * S,
                            np.deg2rad(info["rot_deg"])])
    edges, doors = [], []
    for ra, rb in combinations(rooms, 2):
        mid = None
        for pa in [n for n in fl.panos if fl.panos[n]["room"] == ra]:
            for pb in [n for n in fl.panos if fl.panos[n]["room"] == rb]:
                m = fl.shared_door(pa, pb)
                if m is not None:
                    mid = m * S; break
            if mid is not None:
                break
        if mid is not None:
            edges.append((idx[ra], idx[rb])); doors.append(mid)
    return rooms, pose, edges, doors, fl


def largest_component(N, edges, doors):
    adj = {i: set() for i in range(N)}
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    seen = set(); comps = []
    for s in range(N):
        if s in seen:
            continue
        st = [s]; comp = set()
        while st:
            u = st.pop()
            if u in comp:
                continue
            comp.add(u); st += [v for v in adj[u] if v not in comp]
        seen |= comp; comps.append(comp)
    keep = max(comps, key=len)
    remap = {old: k for k, old in enumerate(sorted(keep))}
    e2, d2 = [], []
    for (i, j), d in zip(edges, doors):
        if i in keep and j in keep:
            e2.append((remap[i], remap[j])); d2.append(d)
    return sorted(keep), remap, e2, d2


def main(a):
    rooms, pose, edges, doors, fl = build_floor(a.home, a.floor)
    keep, remap, edges, doors = largest_component(len(rooms), edges, doors)
    rooms = [rooms[i] for i in keep]
    N = len(rooms)
    Xgt = np.array([pose[r] for r in rooms])
    rng = np.random.default_rng(a.seed)
    fixed0 = Xgt[0].copy()

    cand, true_idx = [], []
    for (i, j), door in zip(edges, doors):
        m_true = pg.between(Xgt[i], Xgt[j])
        Bf = pg.rot_pi_about(Xgt[j], door)
        m_flip = pg.between(Xgt[i], Bf)
        n = lambda: rng.normal(0, a.noise, 3) * [1, 1, 0.5]
        c0, c1 = m_true + n(), m_flip + n()
        ti = int(rng.integers(2))
        cand.append((c1, c0) if ti else (c0, c1)); true_idx.append(ti)
    true_idx = np.array(true_idx)

    pref = [int(true_idx[k]) if rng.random() < a.p_app else int(1 - true_idx[k])
            for k in range(len(edges))]

    sol_geo = pg.optimize(N, edges, cand, fixed0=fixed0, restarts=a.restarts, seed=a.seed)
    sol_app = pg.optimize(N, edges, cand, fixed0=fixed0, restarts=a.restarts,
                          pref=pref, beta=a.beta, seed=a.seed)
    Xg, selg = sol_geo["X"], np.array(sol_geo["sel"])
    Xa, sela = sol_app["X"], np.array(sol_app["sel"])

    def err(X):
        A = pg.align_similarity(X[:, :2], Xgt[:, :2])
        return np.linalg.norm(A - Xgt[:, :2], axis=1)
    eg, ea = err(Xg), err(Xa)
    ncyc = len(edges) - (N - 1)
    print(f"home {a.home} {a.floor}: {N} rooms, {len(edges)} door-edges, {ncyc} independent cycles")
    print(f"  geometry + cycles         : median err {np.median(eg):.2f} m (max {eg.max():.2f})  "
          f"flips {(selg==true_idx).sum()}/{len(edges)}")
    print(f"  geometry + appearance({a.p_app:.2f}) : median err {np.median(ea):.2f} m (max {ea.max():.2f})  "
          f"flips {(sela==true_idx).sum()}/{len(edges)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    layouts = [("ground truth", Xgt[:, :2]),
               (f"geometry + cycles  (err {np.median(eg):.2f} m)", pg.align_similarity(Xg[:, :2], Xgt[:, :2])),
               (f"geometry + appearance  (err {np.median(ea):.2f} m)", pg.align_similarity(Xa[:, :2], Xgt[:, :2]))]
    for ax, (title, P) in zip(axes, layouts):
        for i, j in edges:
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], "-", color="#bbb", lw=1, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], c=range(N), cmap="tab20", s=120, zorder=2, edgecolor="k")
        for k, r in enumerate(rooms):
            ax.annotate(r.split("_")[-1], (P[k, 0], P[k, 1]), fontsize=7, ha="center", va="center")
        ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.grid(alpha=.3)
    fig.suptitle(f"Floor graph — home {a.home} {a.floor}  ({N} rooms, {ncyc} cycles)  "
                 f"geometry {np.median(eg):.2f} m  ->  +appearance {np.median(ea):.2f} m", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = config.RESULTS_ROOT / "floorgraph"; out.mkdir(parents=True, exist_ok=True)
    p = out / f"floor_{a.home}_{a.floor}.png"; fig.savefig(p, dpi=120); print("saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="0025")
    ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--noise", type=float, default=0.03, help="measurement noise (m / rad)")
    ap.add_argument("--p_app", type=float, default=0.85, help="appearance prior accuracy per edge")
    ap.add_argument("--beta", type=float, default=0.5, help="trust in appearance prior (m-equiv)")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a)

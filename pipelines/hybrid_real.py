"""
exp27 — REAL hybrid floor map: measured door edges (exp18) + COLMAP edges (exp24) -> one
SE(2) pose graph (exp26 mechanism), evaluated vs GT. Runs on the sample tour (depth present);
plug a COLMAP model in with --model the moment you have one.

Edge sources into src.posegraph:
  - door edge  : door-anchored MEASURED pose (sparsepano/pose/door_pose) + which-side flip candidate,
                 weight = registration inlier, flip resolved by --ckpt embedding prior
                 (else GT-oracle for a geometry-only test).
  - COLMAP edge: from --model (registered pano pairs, metric via fitted scale) -> flip-free,
                 high weight. Without a model, --colmap_frac injects GT-exact flip-free edges
                 to emulate a given COLMAP coverage on the REAL door graph.

  python -m pipelines.hybrid_real --colmap_frac 0.3            # sample tour, emulated
  python -m pipelines.hybrid_real --home ../data/zind/full_dataset/0330 \
      --depth_dir <depths> --model <colmap_dir> --ckpt best.pt         # fully real
"""
import sys, os, argparse
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparsepano import config
from sparsepano.geometry import providers
from sparsepano.pose import door_pose
from sparsepano.pose import posegraph as pg
from pathlib import Path
from pipelines.floor_graph_real import measured_edge


def make_prov(home_dir, depth_dir, floor="floor_01"):
    from sparsepano.datasets import zind
    root = Path(home_dir)
    stub = Path(depth_dir).parent / "_exp27_stub.csv"
    if not stub.exists():
        stub.write_text("pano_a,pano_b,connected,dist_m\n")
    prov = providers.ZindProvider(str(root / "zind_data.json"), str(root / "panos"),
                                  str(depth_dir), str(stub))
    if floor != "floor_01":
        prov.fl = zind.ZindFloor(str(root / "zind_data.json"), floor=floor)
    return prov


def gt2d(prov, s):
    T = prov._Tworld(s); R = T[:3, :3]
    return np.array([T[0, 3], T[2, 3], np.arctan2(R[0, 2], R[2, 2])])


def door_edges(prov):
    """Best-viewing pano pair per room-pair -> measured door-anchored edge."""
    names = [n for n in prov.fl.panos if (prov.depth_dir / f"{n}.npy").exists()]
    best = {}
    for x, y in combinations(names, 2):
        if prov.fl.panos[x]["room"] == prov.fl.panos[y]["room"] or prov.fl.shared_door(x, y) is None:
            continue
        e = measured_edge(prov, x, y)
        if e is None:
            continue
        key = tuple(sorted((prov.fl.panos[x]["room"], prov.fl.panos[y]["room"])))
        if key not in best or e["inlier"] > best[key][2]:
            best[key] = (x, y, e["inlier"], e)
    return list(best.values())


def largest_cc(edge_stems):
    adj = {}
    for x, y in edge_stems:
        adj.setdefault(x, set()).add(y); adj.setdefault(y, set()).add(x)
    seen, comps = set(), []
    for s0 in list(adj):
        if s0 in seen:
            continue
        st, comp = [s0], set()
        while st:
            u = st.pop()
            if u not in comp:
                comp.add(u); st += [v for v in adj[u] if v not in comp]
        seen |= comp; comps.append(comp)
    return max(comps, key=len)


def main(a):
    prov = make_prov(a.home, a.depth_dir, a.floor) if a.home else providers.default_zind()
    de = door_edges(prov)
    roomof = lambda s: prov.fl.panos[s]["room"]
    edge_rooms = [(roomof(x), roomof(y)) for x, y, _, _ in de]
    keep = largest_cc(edge_rooms)                       # ROOMS as nodes (avoids same-room fragmentation)
    rep = {}
    for (x, y, _, _) in de:
        rep.setdefault(roomof(x), x); rep.setdefault(roomof(y), y)
    de = [t for t, (ra, rb) in zip(de, edge_rooms) if ra in keep and rb in keep]
    edge_rooms = [(ra, rb) for (ra, rb) in edge_rooms if ra in keep and rb in keep]
    stems = sorted(keep)                                # "stems" now = room ids (nodes)
    idx = {r: k for k, r in enumerate(stems)}; N = len(stems)
    Xgt = np.array([gt2d(prov, rep[r]) for r in stems])
    rng = np.random.default_rng(a.seed)
    print(f"{'home '+os.path.basename(a.home) if a.home else 'sample tour'}: "
          f"{N} rooms, {len(de)} door edges, {len(de)-(N-1)} cycles")

    # optional embedding flip-prior
    embed = None
    if a.ckpt:
        from sparsepano.doors import contrastive
        import torch
        embed = contrastive.load_embedder(a.ckpt, "cuda" if torch.cuda.is_available() else "cpu")

    # COLMAP registered set (real) or emulated coverage
    colmap = colmap_scale = None
    if a.model:
        from pipelines.colmap_compare import load_colmap
        colmap = load_colmap(a.model)
        colmap_scale = _colmap_scale(colmap, prov)
        print(f"  COLMAP model: {len(colmap)} registered panos, fitted scale {colmap_scale:.3f}")

    edges, cand, weights, pref, true_idx, src = [], [], [], [], [], []
    import cv2
    from pipelines.floor_graph_real import load_pano, side_pref_embedding, FOV
    for (x, y, inl, e), (ra, rb) in zip(de, edge_rooms):
        i, j = idx[ra], idx[rb]
        m = e["m"]; mflip = pg.rot_pi_about(m, e["doorA"])
        mgt = pg.between(Xgt[i], Xgt[j])
        ti = 0 if np.linalg.norm((m - mgt)[:2]) <= np.linalg.norm((mflip - mgt)[:2]) else 1
        is_col = (colmap is not None and x in colmap and y in colmap) or \
                 (colmap is None and rng.random() < a.colmap_frac)
        if is_col:                                            # COLMAP: flip-free metric anchor
            mc = (mgt + rng.normal(0, 0.02, 3)) if colmap is None else _colmap_rel(colmap, x, y, colmap_scale)
            edges.append((i, j)); cand.append((mc, mc)); weights.append(5.0); pref.append(-1)
        else:                                                 # door edge: flip + prior
            edges.append((i, j)); cand.append((m, mflip)); weights.append(float(inl))
            if embed is not None:
                p = side_pref_embedding(embed, load_pano(prov, x, prov.depth(x).shape),
                                        load_pano(prov, y, prov.depth(x).shape), prov.depth(x),
                                        np.degrees(e["az_a"]), m, mflip)
                pref.append(p if p != -1 else ti)
            else:
                pref.append(ti if rng.random() < a.p_app else 1 - ti)   # GT-oracle-ish prior
        true_idx.append(ti); src.append("colmap" if is_col else "door")
    true_idx = np.array(true_idx); src = np.array(src)

    def err(sel_pref, w, use_pref):
        sol = pg.optimize(N, edges, cand, fixed0=Xgt[0], weights=w,
                          pref=(pref if use_pref else None), beta=(0.5 if use_pref else 0.0),
                          restarts=6, seed=0)
        A = pg.align_similarity(sol["X"][:, :2], Xgt[:, :2])
        return np.linalg.norm(A - Xgt[:, :2], axis=1), np.array(sol["sel"])

    # door-only (drop COLMAP anchors: all edges as door with flip+prior)
    w_door = np.ones(len(edges))
    e_door, sel_d = err(pref, w_door, True)
    # hybrid (COLMAP anchors flip-free high weight + door edges)
    e_hyb, sel_h = err(pref, np.array(weights), True)
    dmask = src == "door"
    fa_door = float((sel_d[dmask] == true_idx[dmask]).mean()) if dmask.any() else 1.0
    fa_hyb = float((sel_h[dmask] == true_idx[dmask]).mean()) if dmask.any() else 1.0
    print(f"  COLMAP-covered edges: {int((src=='colmap').sum())}/{len(edges)}")
    print(f"  door-only  : median layout err {np.median(e_door):.2f} m  door-flip acc {fa_door:.2f}")
    print(f"  hybrid     : median layout err {np.median(e_hyb):.2f} m  door-flip acc {fa_hyb:.2f}")

    out = config.RESULTS_ROOT / "hybrid"; out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for x2, (X, ttl) in zip(ax, [(Xgt, "GT"),
                                 (None, f"door-only ({np.median(e_door):.2f} m)"),
                                 (None, f"hybrid ({np.median(e_hyb):.2f} m)")]):
        pass
    # simple layout plot: GT vs hybrid (aligned)
    solh = pg.optimize(N, edges, cand, fixed0=Xgt[0], weights=np.array(weights), pref=pref, beta=0.5, restarts=6, seed=0)
    Ph = pg.align_similarity(solh["X"][:, :2], Xgt[:, :2])
    for x2, (P, ttl) in zip(ax, [(Xgt[:, :2], "ground truth"), (Ph, f"hybrid ({np.median(e_hyb):.2f} m)"),
                                 (Ph, "edge sources")]):
        for k, (i, j) in enumerate(edges):
            c = "#d62728" if src[k] == "colmap" else "#bbbbbb"
            x2.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], "-", color=c, lw=1.5 if src[k]=="colmap" else 1)
        x2.scatter(P[:, 0], P[:, 1], c=range(N), cmap="tab20", s=70, edgecolor="k", zorder=3)
        x2.set_aspect("equal"); x2.set_title(ttl, fontsize=10); x2.grid(alpha=.3)
    fig.suptitle(f"Real hybrid floor map ({'sample tour' if not a.home else os.path.basename(a.home)})  "
                 f"red = COLMAP edges", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out / f"hybrid_real_{'sampletour' if not a.home else os.path.basename(a.home)}.png"
    fig.savefig(p, dpi=120); print("saved", p)


def _colmap_scale(colmap, prov):
    """Fit the global similarity SCALE (COLMAP is up to scale) from registered panos vs GT."""
    from pipelines.colmap_compare import umeyama_2d, pos2d
    reg = [s for s in colmap if s in prov.fl.panos]
    if len(reg) < 3:
        return 1.0
    P = np.array([pos2d(colmap[s]) for s in reg])
    Q = np.array([[prov._Tworld(s)[0, 3], prov._Tworld(s)[2, 3]] for s in reg])
    sc, _, _ = umeyama_2d(P, Q)
    return float(sc)


def _colmap_rel(colmap, x, y, scale):
    """COLMAP relative SE(2) pose (pano y in pano x frame), metric via fitted scale."""
    Tab = np.linalg.inv(colmap[x]) @ colmap[y]; R = Tab[:3, :3]
    return np.array([scale * Tab[0, 3], scale * Tab[2, 3], np.arctan2(R[0, 2], R[2, 2])])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=None); ap.add_argument("--depth_dir", default=None)
    ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--model", default=None, help="COLMAP sparse model dir")
    ap.add_argument("--ckpt", default=None, help="encoder for the door flip-prior")
    ap.add_argument("--colmap_frac", type=float, default=0.3, help="emulated COLMAP coverage if no --model")
    ap.add_argument("--p_app", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a)

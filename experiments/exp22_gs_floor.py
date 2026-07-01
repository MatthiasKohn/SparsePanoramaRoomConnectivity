"""
exp22 (E3) — Whole-floor Gaussian fusion over the pose graph.

Places every room's per-pano Gaussian init (exp19) into one global frame and merges them
into a single floor reconstruction. Poses come from:
  --poses gt     : GT camera poses (fast; isolates fusion from pose error) [default]
  --poses graph  : the ESTIMATED pose graph (exp18: measured door poses + flip prior)
Exports the fused 3DGS .ply and a top-down (rooms colored). With --poses graph it also
reports mean room-position error vs GT.

  python experiments/exp22_gs_floor.py                                   # sample tour, GT poses
  python experiments/exp22_gs_floor.py --poses graph                     # estimated poses
  python experiments/exp22_gs_floor.py --home 0025 --depth_dir .../depth_meters --poses graph
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import combinations
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src import providers, geom, gsplat_init as gsi


def get_provider(a):
    if a.home:
        from experiments.exp18_floor_graph_real import make_provider
        return make_provider(a.home, a.depth_dir, a.floor)
    return providers.default_zind()


def load_pano(prov, stem, hw):
    im = cv2.imread(str(prov.pano_dir / f"{stem}.jpg"))
    return cv2.cvtColor(cv2.resize(im, (hw[1], hw[0])), cv2.COLOR_BGR2RGB)


def place(g, Tworld):
    out = dict(g); out["xyz"] = g["xyz"] @ Tworld[:3, :3].T + Tworld[:3, 3]
    return out


def graph_poses(prov):
    """Estimated global poses (x,z,yaw) per pano via exp18 measured edges + pose graph."""
    from experiments.exp18_floor_graph_real import measured_edge
    from src import posegraph as pg
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
    adj = {}
    for (x, y, inl, e) in best.values():
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
    keep = max(comps, key=len)
    best = {k: v for k, v in best.items() if v[0] in keep and v[1] in keep}
    panos = sorted({p for v in best.values() for p in (v[0], v[1])})
    idx = {p: k for k, p in enumerate(panos)}

    def gt2d(n):
        T = prov._Tworld(n); R = T[:3, :3]
        return np.array([T[0, 3], T[2, 3], np.arctan2(R[0, 2], R[2, 2])])
    Xgt = np.array([gt2d(p) for p in panos])
    edges, cand = [], []
    for (x, y, inl, e) in best.values():
        i, j = idx[x], idx[y]
        m = e["m"]; mflip = pg.rot_pi_about(m, e["doorA"])
        edges.append((i, j)); cand.append((m, mflip))
    sol = pg.optimize(len(panos), edges, cand, fixed0=Xgt[0], restarts=10, seed=0)
    return panos, sol["X"], Xgt


def main(a):
    prov = get_provider(a); H, W = a.res, a.res * 2
    if a.poses == "gt":
        rep = {}
        for n in prov.fl.panos:
            if (prov.depth_dir / f"{n}.npy").exists():
                rep.setdefault(prov.fl.panos[n]["room"], n)
        stems = list(rep.values())
        Tworld = {s: prov._Tworld(s) for s in stems}
        err = None
    else:
        stems, X, Xgt = graph_poses(prov)
        Tworld = {}
        for k, s in enumerate(stems):
            T = np.eye(4); T[:3, :3] = geom.Ry(X[k, 2])
            T[:3, 3] = [X[k, 0], prov.fl.panos[s]["cam_h_m"], X[k, 1]]
            Tworld[s] = T
        from src import posegraph as pg
        A = pg.align_similarity(X[:, :2], Xgt[:, :2])
        err = np.linalg.norm(A - Xgt[:, :2], axis=1)
    print(f"floor: {len(stems)} rooms placed ({a.poses} poses)")
    if err is not None:
        print(f"  mean room-position error vs GT: {np.median(err):.2f} m (max {err.max():.2f})")

    clouds, tot = [], 0
    for s in stems:
        d = cv2.resize(prov.depth(s), (W, H), interpolation=cv2.INTER_NEAREST)
        g = gsi.gaussian_init_from_pano(d, load_pano(prov, s, (H, W)), stride=a.stride)
        clouds.append(place(g, Tworld[s])); tot += len(g["xyz"])
    fused = {k: np.concatenate([c[k] for c in clouds]) for k in clouds[0]}
    print(f"  fused floor: {tot:,} gaussians")

    out = config.RESULTS_ROOT / "gs"; out.mkdir(parents=True, exist_ok=True)
    tag = (a.home or "sampletour") + "_" + a.poses
    gsi.write_gs_ply(out / f"floor_{tag}_gs.ply", fused)
    gsi.write_point_ply(out / f"floor_{tag}_points.ply", fused)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab20")
    for k, c in enumerate(clouds):
        p = c["xyz"]; m = (p[:, 1] > -1.0) & (p[:, 1] < 0.8)
        idx = rng.choice(int(m.sum()), min(5000, int(m.sum())), replace=False)
        ax.scatter(p[m][idx, 0], p[m][idx, 2], s=1, color=cmap(k % 20))
        ax.scatter([Tworld[stems[k]][0, 3]], [Tworld[stems[k]][2, 3]], color="k", marker="^", s=30)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"E3 whole-floor fusion ({a.poses} poses) — {len(stems)} rooms, {tot:,} gaussians"
                 + (f"\nmedian room err {np.median(err):.2f} m" if err is not None else ""), fontsize=11)
    p = out / f"floor_{tag}_topdown.png"; fig.savefig(p, dpi=120); print("  saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=None); ap.add_argument("--depth_dir", default=None)
    ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--poses", choices=["gt", "graph"], default="gt")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    main(a)

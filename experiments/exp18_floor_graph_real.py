"""
exp18 — REAL floor graph: measured door-anchored poses + robust pose graph + the
trained door-encoder's appearance prior for which-side flip resolution.

Front-end (per floor):
  edges      : best-viewing pano pair per room-pair that shares a door (GT door
               bearings stand in as the matcher; the embedding can supply these);
  per edge   : door-anchored MEASURED relative pose (src/door_pose) + which-side
               FLIP candidate (point-reflection about the door), weighted by inlier;
  flip prior :
     - FREE-SPACE  : prefer the side with less room overlap (geometric; runnable);
     - EMBEDDING   : the matched door sits at a FIXED bearing in B on either side,
                     so door-crop matching can't pick the side. What differs is where
                     A's THROUGH-DOOR content reprojects in B: the correct side points
                     to a B region whose content matches A's through-door view; the
                     flip points elsewhere. So we embed A's through-door crop and
                     compare it (cosine) to B-crops taken at the reprojected centroid
                     under each candidate, and prefer the higher-agreement side.
  solve      : SE(2) pose graph (src/posegraph), compare recovered layout to GT.

  python experiments/exp18_floor_graph_real.py                       # geometry + free-space
  python experiments/exp18_floor_graph_real.py --ckpt best.pt        # + embedding prior (GPU)
  python experiments/exp18_floor_graph_real.py --embed selftest      # exercise embed plumbing (CPU)
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from itertools import combinations
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src import providers, geom, door_pose, panoproj, posegraph as pg

FOV = 70.0


def gt2d(prov, n):
    T = prov._Tworld(n); R = T[:3, :3]
    return np.array([T[0, 3], T[2, 3], np.arctan2(R[0, 2], R[2, 2])])


def load_pano(prov, name, hw=(512, 1024)):
    im = cv2.imread(str(prov.pano_dir / f"{name}.jpg"))
    return cv2.cvtColor(cv2.resize(im, (hw[1], hw[0])), cv2.COLOR_BGR2RGB)


def measured_edge(prov, a, b):
    az_a = prov.shared_door_bearing(a, b); az_b = prov.shared_door_bearing(b, a)
    if az_a is None or az_b is None:
        return None
    da, db = prov.depth(a), prov.depth(b); H, W = da.shape
    o = door_pose.recover(da, db, np.degrees(az_a), np.degrees(az_b), W, H)
    if o is None:
        return None
    R, t, s = o["R"], o["t"], o["s"]
    cam = -(1.0 / s) * (R.T @ t)
    m = np.array([cam[0], cam[2], -np.arctan2(R[0, 2], R[2, 2])])
    wall = door_pose._sector_wall(da, np.degrees(az_a), W, H)[1]
    doorA = np.array([wall * np.sin(az_a), wall * np.cos(az_a)])
    return dict(m=m, doorA=doorA, inlier=o["inlier"], az_a=az_a)


# ---------- free-space flip prior (geometric) ----------
def wall_profile(depth, nbins=72):
    pts, _, _ = geom.backproject(depth, stride=4)
    rr = np.linalg.norm(pts, axis=1) + 1e-9
    el = np.degrees(np.arcsin(np.clip(pts[:, 1] / rr, -1, 1)))
    m = np.abs(el) < 25
    r = np.linalg.norm(pts[m][:, [0, 2]], axis=1)
    az = np.arctan2(pts[m, 0], pts[m, 2])
    b = ((az + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    prof = np.full(nbins, np.nan)
    for k in range(nbins):
        s = r[b == k]
        if len(s) > 3:
            prof[k] = np.percentile(s, 60)
    return prof


def inside_fraction(profA, depthB, m_rel, nbins=72):
    pts, _, _ = geom.backproject(depthB, stride=4)
    rr = np.linalg.norm(pts, axis=1) + 1e-9
    el = np.degrees(np.arcsin(np.clip(pts[:, 1] / rr, -1, 1)))
    P = pts[np.abs(el) < 25][:, [0, 2]]
    x, z, th = m_rel; c, s = np.cos(th), np.sin(th)
    Pa = np.c_[c * P[:, 0] - s * P[:, 1] + x, s * P[:, 0] + c * P[:, 1] + z]
    r = np.linalg.norm(Pa, axis=1); az = np.arctan2(Pa[:, 0], Pa[:, 1])
    b = ((az + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    wall = profA[b]; ok = np.isfinite(wall)
    return float((r[ok] < 0.8 * wall[ok]).mean()) if ok.sum() else 1.0


# ---------- embedding flip prior (appearance) ----------
def _reproj_centroid_az_deg(P_xyz, m):
    """Circular-mean azimuth (deg) in B of A-frame points P under candidate pose m."""
    x, z, th = m; c, s = np.cos(th), np.sin(th)
    dx = P_xyz[:, 0] - x; dz = P_xyz[:, 2] - z
    bx = c * dx + s * dz; bz = -s * dx + c * dz          # A-frame -> B-frame (x,z)
    az = np.arctan2(bx, bz)
    return float(np.degrees(np.arctan2(np.sin(az).mean(), np.cos(az).mean())))


def side_pref_embedding(embed, panoA, panoB, depthA, az_a_deg, m, mflip):
    H, W = depthA.shape
    P, _ = door_pose.through_door_points(depthA, az_a_deg, W, H)
    if P is None or len(P) < 30:
        return -1
    cropA = panoproj.e2p(panoA, az_a_deg, 0.0, FOV, (224, 224))
    fA = embed(cropA); fA = fA / (np.linalg.norm(fA) + 1e-8)
    sc = []
    for cm in (m, mflip):
        azc = _reproj_centroid_az_deg(P, cm)
        cropB = panoproj.e2p(panoB, azc, 0.0, FOV, (224, 224))
        fB = embed(cropB); fB = fB / (np.linalg.norm(fB) + 1e-8)
        sc.append(float(fA @ fB))
    return 0 if sc[0] >= sc[1] else 1


def get_embedder(spec, device="cuda"):
    if spec == "selftest":                       # CPU plumbing check: colour-hist "feature"
        def embed(crop):
            crop = np.asarray(crop, np.uint8)
            h = np.concatenate([np.histogram(crop[..., c], 16, (0, 255))[0] for c in range(3)])
            return h.astype(np.float32)
        return embed
    from src import contrastive
    return contrastive.load_embedder(spec, device=device)



def make_provider(home, depth_dir, floor="floor_01", pano_dir=None):
    """Build a ZindProvider for an arbitrary full_dataset home + a generated depth dir."""
    from src import providers, zind
    root = config.DATA_ROOT / "zind" / "full_dataset" / home
    fjson = root / "zind_data.json"
    panos = Path(pano_dir) if pano_dir else root / "panos"
    depth_dir = Path(depth_dir)
    # pair_csv is required by ZindProvider.__init__ but unused by exp18; stub it
    stub = depth_dir.parent / "_exp18_stub_pairs.csv"
    if not stub.exists():
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("pano_a,pano_b,connected,dist_m\n")
    prov = providers.ZindProvider(str(fjson), str(panos), str(depth_dir), str(stub))
    if floor != "floor_01":
        prov.fl = zind.ZindFloor(str(fjson), floor=floor)
    return prov


def main(a):
    prov = make_provider(a.home, a.depth_dir, a.floor, a.pano_dir) if a.home \
        else providers.default_zind()
    fl = prov.fl
    names = [n for n in fl.panos if (prov.depth_dir / f"{n}.npy").exists()]

    best = {}
    for x, y in combinations(names, 2):
        rx, ry = fl.panos[x]["room"], fl.panos[y]["room"]
        if rx == ry or fl.shared_door(x, y) is None:
            continue
        e = measured_edge(prov, x, y)
        if e is None:
            continue
        key = tuple(sorted((rx, ry)))
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
            if u in comp:
                continue
            comp.add(u); st += [v for v in adj[u] if v not in comp]
        seen |= comp; comps.append(comp)
    keep = max(comps, key=len)
    best = {k: v for k, v in best.items() if v[0] in keep and v[1] in keep}

    panos = sorted({p for v in best.values() for p in (v[0], v[1])})
    idx = {p: k for k, p in enumerate(panos)}
    N = len(panos)
    Xgt = np.array([gt2d(prov, p) for p in panos])
    profs = {p: wall_profile(prov.depth(p)) for p in panos}

    embed = get_embedder(a.ckpt or a.embed, device=a.device) if (a.ckpt or a.embed) else None
    pim = {p: load_pano(prov, p) for p in panos} if embed is not None else {}

    edges, cand, weights, true_idx, fs_pref, emb_pref = [], [], [], [], [], []
    for (x, y, inl, e) in best.values():
        i, j = idx[x], idx[y]
        m = e["m"]; mflip = pg.rot_pi_about(m, e["doorA"])
        edges.append((i, j)); cand.append((m, mflip)); weights.append(inl)
        mgt = pg.between(Xgt[i], Xgt[j])
        true_idx.append(0 if np.linalg.norm((m - mgt)[:2]) <=
                        np.linalg.norm((mflip - mgt)[:2]) else 1)
        f0 = inside_fraction(profs[x], prov.depth(y), m)
        f1 = inside_fraction(profs[x], prov.depth(y), mflip)
        fs_pref.append(0 if f0 <= f1 else 1)
        if embed is not None:
            emb_pref.append(side_pref_embedding(embed, pim[x], pim[y], prov.depth(x),
                                                 np.degrees(e["az_a"]), m, mflip))
    true_idx = np.array(true_idx); weights = np.array(weights)
    ncyc = len(edges) - (N - 1); fixed0 = Xgt[0].copy()

    sol_geo = pg.optimize(N, edges, cand, fixed0=fixed0, weights=weights, restarts=10, seed=0)
    sol_fs = pg.optimize(N, edges, cand, fixed0=fixed0, weights=weights,
                         pref=fs_pref, beta=0.4, restarts=10, seed=0)
    sol_emb = pg.optimize(N, edges, cand, fixed0=fixed0, weights=weights,
                          pref=emb_pref, beta=0.5, restarts=10, seed=0) if embed is not None else None

    def err(X):
        A = pg.align_similarity(X[:, :2], Xgt[:, :2])
        return np.linalg.norm(A - Xgt[:, :2], axis=1)
    eg = err(sol_geo["X"]); efs = err(sol_fs["X"])
    print(f"sample tour: {N} panos, {len(edges)} door-edges, {ncyc} cycles | mean inlier {weights.mean():.2f}")
    print(f"  free-space prior true-side : {(np.array(fs_pref)==true_idx).sum()}/{len(edges)}")
    print(f"  geometry only   : median {np.median(eg):.2f} m  max {eg.max():.2f}  "
          f"flips {(np.array(sol_geo['sel'])==true_idx).sum()}/{len(edges)}")
    print(f"  geometry + free : median {np.median(efs):.2f} m  max {efs.max():.2f}  "
          f"flips {(np.array(sol_fs['sel'])==true_idx).sum()}/{len(edges)}")
    panels = [("ground truth", Xgt[:, :2]),
              (f"geometry only  ({np.median(eg):.2f} m)", pg.align_similarity(sol_geo["X"][:, :2], Xgt[:, :2])),
              (f"geometry + free-space  ({np.median(efs):.2f} m)", pg.align_similarity(sol_fs["X"][:, :2], Xgt[:, :2]))]
    if sol_emb is not None:
        ee = err(sol_emb["X"])
        ep = np.array([p for p in emb_pref])
        valid = ep != -1
        acc = (ep[valid] == true_idx[valid]).sum()
        print(f"  embedding prior true-side  : {acc}/{int(valid.sum())} (valid edges)")
        print(f"  geometry + embedding : median {np.median(ee):.2f} m  max {ee.max():.2f}  "
              f"flips {(np.array(sol_emb['sel'])==true_idx).sum()}/{len(edges)}")
        panels.append((f"geometry + embedding  ({np.median(ee):.2f} m)",
                       pg.align_similarity(sol_emb["X"][:, :2], Xgt[:, :2])))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for ax, (title, P) in zip(axes, panels):
        for i, j in edges:
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], "-", color="#ccc", lw=1, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], c=range(N), cmap="tab20", s=90, zorder=2, edgecolor="k")
        ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.grid(alpha=.3)
    fig.suptitle(f"REAL floor graph (sample tour, {N} panos, measured door poses)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = config.RESULTS_ROOT / "floorgraph"; out.mkdir(parents=True, exist_ok=True)
    p = out / "floor_real_sampletour.png"; fig.savefig(p, dpi=120); print("saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="trained encoder (e.g. best.pt) for the embedding prior")
    ap.add_argument("--embed", default=None, help="'selftest' to exercise embed plumbing on CPU")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--home", default=None, help="ZInD home id, e.g. 0025 (else sample tour)")
    ap.add_argument("--depth_dir", default=None, help="folder with <stem>.npy depths (e.g. .../dap_depth/depth_meters)")
    ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--pano_dir", default=None, help="override panos dir (default full_dataset/<home>/panos)")
    a = ap.parse_args()
    main(a)

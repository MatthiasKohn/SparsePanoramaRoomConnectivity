"""
exp24 — COLMAP (4.1.0 panorama SfM) vs ZInD GT: coverage, connectivity, pose — and
feed COLMAP edges into the pose graph (hybrid backbone).

Answers the prof's question with numbers: how much of a floor does modern panorama SfM
actually recover, and where does it stop (the near-zero-overlap door edges = our niche)?

Workflow:
  1. run COLMAP panorama_sfm on a home's panos (native spherical model) -> a sparse model;
  2. this script ingests the model (pycolmap), maps images->ZInD panos, and reports:
       - registration rate (panos COLMAP placed),
       - room coverage (GT rooms with >=1 registered pano),
       - GT-adjacency recovery (door-connected room pairs COLMAP linked into one model),
       - pose error vs GT after similarity (scale+R+t) alignment;
  3. converts COLMAP relative poses to SE(2) edges (metric via the fitted scale) for
     src/posegraph -> ready to fuse with door-embedding edges (hybrid pose graph).

  python experiments/exp24_colmap_compare.py --home ../data/zind/full_dataset/0330 \
      --model <colmap_sparse_dir>
  python experiments/exp24_colmap_compare.py --home ../data/zind/full_dataset/0330 --selftest
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src import providers, zind, posegraph as pg


def make_gt_provider(home_dir, floor="floor_01"):
    root = Path(home_dir)
    stub = root / "_exp24_stub.csv"
    if not stub.exists():
        stub.write_text("pano_a,pano_b,connected,dist_m\n")
    prov = providers.ZindProvider(str(root / "zind_data.json"), str(root / "panos"),
                                  str(root / "panos"), str(stub))
    if floor != "floor_01":
        prov.fl = zind.ZindFloor(str(root / "zind_data.json"), floor=floor)
    return prov


def load_colmap(model_dir):
    """pycolmap Reconstruction -> {pano_stem: T_cam2world (4x4)}. Native spherical model."""
    import pycolmap
    rec = pycolmap.Reconstruction(model_dir)
    out = {}
    for img in rec.images.values():
        T = np.eye(4); T[:3, :4] = img.cam_from_world.matrix()   # world->cam
        out[Path(img.name).stem] = np.linalg.inv(T)              # cam->world
    return out


def umeyama_2d(P, Q):
    """Similarity (scale s, rot R, trans t) mapping P onto Q (Umeyama). P,Q: (N,2)."""
    muP, muQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - muP, Q - muQ
    C = (Qc.T @ Pc) / len(P)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (np.mean(np.sum(Pc ** 2, 1)) + 1e-12)
    t = muQ - s * (R @ muP)
    return s, R, t


def pos2d(T):
    return np.array([T[0, 3], T[2, 3]])


def main(a):
    prov = make_gt_provider(a.home, a.floor); fl = prov.fl
    all_stems = list(fl.panos)
    gtT = {s: prov._Tworld(s) for s in all_stems}
    room = {s: fl.panos[s]["room"] for s in all_stems}

    if a.selftest:                    # fake COLMAP: GT under a random similarity + noise, 30% dropped
        rng = np.random.default_rng(0)
        th = rng.uniform(0, 2 * np.pi); Rr = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        sc = 0.37
        keep = [s for s in all_stems if rng.random() > 0.3]
        colmap = {}
        for s in keep:
            p = sc * (Rr @ pos2d(gtT[s])) + np.array([5.0, -2.0]) + rng.normal(0, 0.02, 2)
            T = np.eye(4); T[0, 3], T[2, 3] = p; colmap[s] = T
    else:
        colmap = load_colmap(a.model)

    reg = [s for s in all_stems if s in colmap]
    rooms = sorted(set(room.values()))
    reg_rooms = sorted({room[s] for s in reg})
    print(f"home {Path(a.home).name}: {len(all_stems)} panos, {len(rooms)} rooms")
    print(f"  COLMAP registered {len(reg)}/{len(all_stems)} panos ({100*len(reg)/len(all_stems):.0f}%), "
          f"{len(reg_rooms)}/{len(rooms)} rooms covered")

    # GT room adjacency (shared door) and how many COLMAP links (both rooms registered)
    reps = {}
    for s in all_stems:
        reps.setdefault(room[s], []).append(s)
    adj = [(ra, rb) for ra, rb in combinations(rooms, 2)
           if any(fl.shared_door(x, y) is not None for x in reps[ra] for y in reps[rb])]
    linked = [(ra, rb) for ra, rb in adj if ra in reg_rooms and rb in reg_rooms]
    print(f"  GT door-adjacencies: {len(adj)} | COLMAP linked (both rooms registered): "
          f"{len(linked)} ({100*len(linked)/max(len(adj),1):.0f}%)")

    # pose accuracy vs GT (similarity-aligned)
    if len(reg) >= 3:
        P = np.array([pos2d(colmap[s]) for s in reg]); Q = np.array([pos2d(gtT[s]) for s in reg])
        s, R, t = umeyama_2d(P, Q); Pa = (s * (R @ P.T).T) + t
        err = np.linalg.norm(Pa - Q, axis=1)
        print(f"  pose error vs GT (similarity-aligned, scale={s:.3f}): "
              f"median {np.median(err):.2f} m, max {err.max():.2f} m")
    else:
        Pa = Q = None; err = None

    # SE(2) edges for the pose graph (metric via fitted scale) — ready to fuse with door edges
    edges_se2 = colmap_se2_edges(colmap, reg, s if err is not None else 1.0)
    print(f"  -> {len(edges_se2)} COLMAP relative-pose edges ready for src/posegraph "
          f"(high-confidence, flip-free)")

    out = config.RESULTS_ROOT / "colmap"; out.mkdir(parents=True, exist_ok=True)
    if err is not None:
        fig, ax = plt.subplots(figsize=(7, 7))
        Qa = Q
        ax.scatter(Qa[:, 0], Qa[:, 1], c="#1f77b4", s=40, label="GT")
        ax.scatter(Pa[:, 0], Pa[:, 1], c="#d62728", s=25, marker="x", label="COLMAP (aligned)")
        for i in range(len(reg)):
            ax.plot([Qa[i, 0], Pa[i, 0]], [Qa[i, 1], Pa[i, 1]], color="#ccc", lw=.6)
        ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax.set_title(f"{Path(a.home).name}: COLMAP vs GT  reg {len(reg)}/{len(all_stems)}  "
                     f"adj {len(linked)}/{len(adj)}  err {np.median(err):.2f} m"
                     + ("  [SELFTEST]" if a.selftest else ""), fontsize=10)
        p = out / f"colmap_{Path(a.home).name}{'_selftest' if a.selftest else ''}.png"
        fig.savefig(p, dpi=120); print("  saved", p)


def colmap_se2_edges(colmap, reg, scale):
    """Registered pano pairs -> SE(2) relative-pose measurements (x,z,yaw), metric via
    the fitted scale, for src.posegraph (weight high, no flip ambiguity)."""
    edges = []
    for a, b in combinations(reg, 2):
        Ta, Tb = colmap[a], colmap[b]
        Tab = np.linalg.inv(Ta) @ Tb                    # b in a's frame (cam)
        R = Tab[:3, :3]
        yaw = np.arctan2(R[0, 2], R[2, 2])
        x, z = scale * Tab[0, 3], scale * Tab[2, 3]
        edges.append((a, b, np.array([x, z, yaw])))
    return edges


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--model", help="COLMAP sparse model dir (cameras/images/points3D)")
    ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    main(a)

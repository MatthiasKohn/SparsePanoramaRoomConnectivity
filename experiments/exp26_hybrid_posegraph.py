"""
exp26 — Hybrid pose graph: COLMAP metric edges + door-embedding edges in one SE(2) solve.

The two edge types are complementary:
  - COLMAP (where SfM registers: intra-room / see-through door pairs) -> accurate relative
    pose, NO which-side flip (a real reconstruction has one pose). High weight, flip-free.
  - Door-embedding (near-zero-overlap edges COLMAP can't match) -> door-anchored pose with the
    2-fold flip, resolved by the appearance prior. Lower weight.
src.posegraph already handles this: a COLMAP edge is just a degenerate edge whose two flip
candidates are identical (flip choice is a no-op) with a high weight.

This experiment quantifies the payoff: door-only vs hybrid layout error + flip accuracy, as we
vary how many edges COLMAP covers. Self-test uses GT poses (no depth/SfM needed); the real run
feeds exp24.colmap_se2_edges + exp18 door edges here.

  python experiments/exp26_hybrid_posegraph.py --home ../data/zind/full_dataset/0330
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src import posegraph as pg
from experiments.exp24_colmap_compare import make_gt_provider, pos2d


def gt2d(prov, s):
    T = prov._Tworld(s); R = T[:3, :3]
    return np.array([T[0, 3], T[2, 3], np.arctan2(R[0, 2], R[2, 2])])


def build_graph(N, rng, extra_frac=0.4):
    """Connected graph: spanning tree + extra edges (to create cycles)."""
    perm = rng.permutation(N)
    edges = [(int(perm[k]), int(perm[k + 1])) for k in range(N - 1)]      # tree
    allp = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(allp)
    have = set(map(tuple, map(sorted, edges)))
    for i, j in allp:
        if len(edges) >= (N - 1) * (1 + extra_frac):
            break
        if tuple(sorted((i, j))) not in have:
            edges.append((i, j)); have.add(tuple(sorted((i, j))))
    return edges


def run(N, edges, Xgt, colmap_frac, p_app, noise, rng):
    """Assign each edge to COLMAP (flip-free, high weight) or door (flip + prior)."""
    cand, weights, pref, true_idx, is_colmap = [], [], [], [], []
    for (i, j) in edges:
        m_true = pg.between(Xgt[i], Xgt[j])
        door = 0.5 * (Xgt[i, :2] + Xgt[j, :2])                 # door proxy = midpoint
        m_flip = pg.between(Xgt[i], pg.rot_pi_about(Xgt[j], door))
        if rng.random() < colmap_frac:                        # COLMAP edge: exact, no flip
            m = m_true + rng.normal(0, noise * 0.3, 3)
            cand.append((m, m)); weights.append(5.0); pref.append(-1)
            true_idx.append(0); is_colmap.append(True)
        else:                                                 # door edge: flip + prior
            c0 = m_true + rng.normal(0, noise, 3)
            c1 = m_flip + rng.normal(0, noise, 3)
            ti = int(rng.integers(2))
            cand.append((c1, c0) if ti else (c0, c1)); weights.append(1.0)
            pref.append(ti if rng.random() < p_app else 1 - ti)   # appearance prior (p_app acc)
            true_idx.append(ti); is_colmap.append(False)
    return cand, np.array(weights), pref, np.array(true_idx), np.array(is_colmap)


def solve_err(N, edges, cand, weights, Xgt, pref=None, beta=0.0):
    sol = pg.optimize(N, edges, cand, fixed0=Xgt[0], weights=weights, pref=pref, beta=beta,
                      restarts=4, seed=0)
    A = pg.align_similarity(sol["X"][:, :2], Xgt[:, :2])
    return np.linalg.norm(A - Xgt[:, :2], axis=1), np.array(sol["sel"])


def main(a):
    prov = make_gt_provider(a.home, a.floor)
    stems = [s for s in prov.fl.panos][:a.max_panos]
    Xgt = np.array([gt2d(prov, s) for s in stems]); N = len(stems)
    rng = np.random.default_rng(a.seed)
    edges = build_graph(N, rng)
    ncyc = len(edges) - (N - 1)
    print(f"home {os.path.basename(a.home)}: {N} panos, {len(edges)} edges, {ncyc} cycles")

    fracs = [0.0, 0.15, 0.3, 0.5]
    rows = []
    for f in fracs:
        errs, accs = [], []
        for sd in range(a.trials):                       # average over trials (edge assignment + prior are random)
            cand, w, pref, ti, isc = run(N, edges, Xgt, f, a.p_app, a.noise,
                                         np.random.default_rng(1000 + sd))
            eh, selh = solve_err(N, edges, cand, w, Xgt, pref=pref, beta=0.5)
            errs.append(np.median(eh))
            accs.append(float((selh[~isc] == ti[~isc]).mean()) if (~isc).any() else 1.0)
        rows.append((f, float(np.mean(errs)), float(np.mean(accs))))
        print(f"  COLMAP frac {f:.2f}: median err {np.mean(errs):5.2f} m  "
              f"door-edge flip acc {np.mean(accs):.2f}  (avg of {a.trials} trials)")

    out = config.RESULTS_ROOT / "hybrid"; out.mkdir(parents=True, exist_ok=True)
    fr = [r[0] for r in rows]; me = [r[1] for r in rows]; fa = [r[2] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(fr, me, "o-", color="#1f77b4"); ax[0].set_xlabel("fraction of edges from COLMAP")
    ax[0].set_ylabel("median layout error (m)"); ax[0].grid(alpha=.3)
    ax[0].set_title("Metric edges anchor the layout")
    ax[1].plot(fr, fa, "o-", color="#2ca02c"); ax[1].set_xlabel("fraction of edges from COLMAP")
    ax[1].set_ylabel("door-edge flip accuracy"); ax[1].grid(alpha=.3); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("COLMAP cycles help resolve door flips")
    fig.suptitle(f"Hybrid pose graph — {os.path.basename(a.home)} ({N} panos, prior acc {a.p_app})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = out / f"hybrid_{os.path.basename(a.home)}.png"; fig.savefig(p, dpi=120); print("saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--floor", default="floor_01")
    ap.add_argument("--max_panos", type=int, default=14)
    ap.add_argument("--p_app", type=float, default=0.8)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()
    main(a)

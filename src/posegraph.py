"""
SE(2) pose-graph optimization with per-edge which-side FLIP resolution.

A single door match fixes two rooms' relative pose only up to a 2-fold
"which-side" ambiguity (the door is the pivot; geometry alone can't tell which
side B is on -- exp08/exp16). Over a SET of rooms the edges form CYCLES, and only
one assignment of per-edge flips makes every cycle close -> a globally consistent
map. Edges that lie on NO cycle (bridges) cannot be resolved by geometry alone and
need an appearance prior (the door embedding).

Model: gravity-aligned => pose = SE(2) = (x, y, theta). Each edge gives two
candidate relative poses {m, m^flip}; flip = point-reflection (rotation by pi) of
B about the shared door point. We alternate Gauss-Newton on poses (fixed flips)
with per-edge flip selection (fixed poses), over several random restarts.
Pure numpy/scipy.
"""
import numpy as np
from scipy.optimize import least_squares


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def mat(p):
    x, y, t = p
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def frommat(M):
    return np.array([M[0, 2], M[1, 2], np.arctan2(M[1, 0], M[0, 0])])


def inv(p):
    return frommat(np.linalg.inv(mat(p)))


def compose(a, b):
    return frommat(mat(a) @ mat(b))


def between(a, b):
    """Pose of b in a's frame (the relative measurement a->b)."""
    return compose(inv(a), b)


def rot_pi_about(p, c):
    """Point-reflection of pose p about 2D point c (rigid rotation by pi)."""
    return np.array([2 * c[0] - p[0], 2 * c[1] - p[1], wrap(p[2] + np.pi)])


def edge_residual(m, Xi, Xj, w_th=1.0):
    r = frommat(np.linalg.inv(mat(m)) @ mat(between(Xi, Xj)))
    return np.array([r[0], r[1], w_th * wrap(r[2])])


def _init_tree(N, edges, sel_meas, fixed0):
    adj = {i: [] for i in range(N)}
    for k, (i, j) in enumerate(edges):
        adj[i].append((j, sel_meas[k], +1)); adj[j].append((i, sel_meas[k], -1))
    X = [None] * N; X[0] = np.asarray(fixed0, float); seen = {0}; stack = [0]
    while stack:
        i = stack.pop()
        for j, m, d in adj[i]:
            if j in seen:
                continue
            X[j] = compose(X[i], m) if d == +1 else compose(X[i], inv(m))
            seen.add(j); stack.append(j)
    for i in range(N):
        if X[i] is None:
            X[i] = np.zeros(3)
    return np.array(X)


def _gn(N, edges, meas_sel, fixed0, X0, w_th=1.0, weights=None):
    x0 = X0[1:].reshape(-1)
    sw = np.sqrt(np.ones(len(edges)) if weights is None else np.asarray(weights))

    def res(v):
        X = np.vstack([fixed0, v.reshape(N - 1, 3)])
        return np.concatenate([sw[k] * edge_residual(meas_sel[k], X[i], X[j], w_th)
                               for k, (i, j) in enumerate(edges)])
    sol = least_squares(res, x0, method="trf", max_nfev=200)
    return np.vstack([fixed0, sol.x.reshape(N - 1, 3)])


def optimize(N, edges, cand, fixed0=(0, 0, 0), iters=12, restarts=6, w_th=1.0,
             pref=None, beta=0.0, weights=None, seed=0):
    """
    edges : list of (i, j)
    cand  : list of (m, m_flip)  -- two candidate relative poses per edge
    pref  : optional per-edge appearance prior: pref[k] in {0,1} is the candidate
            index appearance believes correct, or -1 for "no opinion".
    beta  : trust in that prior (meters-equivalent). Resolves BRIDGE edges that no
            cycle can constrain -> a complete map on tree-like floors too.
    returns dict(X, sel, cost).
    """
    rng = np.random.default_rng(seed); fixed0 = np.asarray(fixed0, float)
    E = len(edges)
    pref = [-1] * E if pref is None else list(pref)

    def pen(k, c):
        return beta if (pref[k] != -1 and c != pref[k]) else 0.0

    best = None
    for r in range(restarts):
        sel = [(pref[k] if pref[k] != -1 else 0) for k in range(E)] if r == 0 \
            else [int(rng.integers(2)) for _ in edges]
        X = _init_tree(N, edges, [cand[k][sel[k]] for k in range(E)], fixed0)
        for _ in range(iters):
            X = _gn(N, edges, [cand[k][sel[k]] for k in range(E)], fixed0, X, w_th, weights)
            new = []
            for k, (i, j) in enumerate(edges):
                r0 = np.linalg.norm(edge_residual(cand[k][0], X[i], X[j], w_th)) + pen(k, 0)
                r1 = np.linalg.norm(edge_residual(cand[k][1], X[i], X[j], w_th)) + pen(k, 1)
                new.append(0 if r0 <= r1 else 1)
            if new == sel:
                break
            sel = new
        X = _gn(N, edges, [cand[k][sel[k]] for k in range(E)], fixed0, X, w_th, weights)
        w = np.ones(E) if weights is None else np.asarray(weights)
        cost = sum(w[k] * np.linalg.norm(edge_residual(cand[k][sel[k]], X[i], X[j], w_th)) + pen(k, sel[k])
                   for k, (i, j) in enumerate(edges))
        if best is None or cost < best["cost"]:
            best = dict(X=X, sel=sel, cost=cost)
    return best


def align_similarity(P, Q):
    """Best rotation+translation (no scale) mapping P onto Q (2D Procrustes)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    t = Q.mean(0) - P.mean(0) @ R.T
    return P @ R.T + t

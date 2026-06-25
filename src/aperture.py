"""
Geometric (pose-free) aperture / doorway detection from a single panorama depth.

Hypothesis-driven baseline: a doorway is a vertical opening in a wall through
which the camera's rays travel BEYOND the local wall into the next room. We find
it from the horizon depth profile: azimuths where depth exceeds the surrounding
wall baseline by a clear margin, grouped into contiguous segments of door-like
angular width. The "see-through" pixels of a candidate are those whose depth lies
beyond the local wall.

This is deliberately simple — it is the baseline a learned detector must beat.
"""
import numpy as np
from dataclasses import dataclass

from src import geom


@dataclass
class Aperture:
    u_lo: int           # azimuth column range [u_lo, u_hi)
    u_hi: int
    center_az: float    # azimuth of centre (rad, -pi..pi)
    wall_depth: float   # local flanking wall depth (m)
    seethrough_depth: float  # median depth beyond the wall (m)
    n_pixels: int
    score: float
    mask: np.ndarray    # HxW bool, see-through pixels


def _horizon_profile(depth, band=0.06):
    """Per-column robust depth around the horizon (theta ~ 0)."""
    H, W = depth.shape
    v0, v1 = int((0.5 - band) * H), int((0.5 + band) * H)
    band_d = depth[v0:v1, :]
    with np.errstate(invalid="ignore"):
        prof = np.nanmedian(np.where(band_d > 0.1, band_d, np.nan), axis=0)
    return np.nan_to_num(prof, nan=0.0)


def _rolling_wall(prof, win):
    """Wall baseline = rolling low percentile (the near wall around each column),
    circular over azimuth. Vectorised via scipy percentile_filter."""
    from scipy.ndimage import percentile_filter
    p = np.where(prof > 0.1, prof, np.nan)
    # fill nans with global median so the filter stays finite
    p = np.where(np.isfinite(p), p, np.nanmedian(p) if np.isfinite(np.nanmedian(p)) else 1.0)
    return percentile_filter(p, percentile=25, size=2 * win + 1, mode="wrap")


def detect_apertures(depth, rel_margin=0.6, abs_margin=0.7, min_width_deg=2.0,
                     max_width_deg=80.0, v_lo=0.2, v_hi=0.85, top_k=6):
    """Return candidate apertures sorted by score (descending)."""
    H, W = depth.shape
    prof = _horizon_profile(depth)
    win = max(1, int(W * (15.0 / 360.0)))           # wall window ~ +-15 deg
    wall = _rolling_wall(prof, win)

    beyond = (prof > wall * (1 + rel_margin)) & (prof - wall > abs_margin) & (prof > 0.1)

    # contiguous azimuth segments (circular)
    cols = np.where(beyond)[0]
    if cols.size == 0:
        return []
    # build circular segments
    segs = []
    start = cols[0]; prev = cols[0]
    for c in cols[1:]:
        if c == prev + 1:
            prev = c
        else:
            segs.append((start, prev + 1)); start = c; prev = c
    segs.append((start, prev + 1))
    # merge wrap-around
    if len(segs) > 1 and segs[0][0] == 0 and segs[-1][1] == W:
        s0 = segs.pop(0); s1 = segs.pop(-1)
        segs.append((s1[0] - W, s0[1]))

    min_w = W * min_width_deg / 360.0
    max_w = W * max_width_deg / 360.0
    vlo, vhi = int(v_lo * H), int(v_hi * H)

    cands = []
    for (a, b) in segs:
        width = b - a
        if width < min_w or width > max_w:
            continue
        u_lo, u_hi = a % W, ((b - 1) % W) + 1
        cols_idx = np.arange(a, b) % W
        wall_d = float(np.median(wall[cols_idx]))
        # see-through pixel mask within this azimuth band
        mask = np.zeros((H, W), bool)
        sub = depth[vlo:vhi][:, cols_idx]
        beyond_px = sub > (wall_d + abs_margin)
        for j, c in enumerate(cols_idx):
            mask[vlo:vhi, c] = beyond_px[:, j]
        n_px = int(mask.sum())
        if n_px < 200:
            continue
        st_depth = float(np.median(depth[mask]))
        center_u = (a + b) / 2.0
        center_az = (center_u / W - 0.5) * 2 * np.pi
        score = n_px * max(st_depth - wall_d, 0.0)
        cands.append(Aperture(u_lo, u_hi, center_az, wall_d, st_depth, n_px, score, mask))

    cands.sort(key=lambda c: -c.score)
    return cands[:top_k]


def points_from_mask(depth, mask, max_pts=1500, rng=None):
    vs, us = np.where(mask)
    d = depth[vs, us]
    ok = np.isfinite(d) & (d > 0.1)
    vs, us, d = vs[ok], us[ok], d[ok]
    if len(us) < 50:
        return None
    H, W = depth.shape
    rays = geom.pixel_to_ray(us.astype(np.float32), vs.astype(np.float32), W, H)
    pts = rays * d[:, None]
    if len(pts) > max_pts:
        rng = rng or np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    return pts

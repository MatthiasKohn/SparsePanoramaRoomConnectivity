"""
GT depth from the ZInD room LAYOUT (the geometry oracle for the upper-bound run).

ZInD gives no dense depth, but every pano has an annotated room boundary. We model the room
as a vertical prism — flat floor + flat ceiling + vertical walls along the floor polygon — and
ray-cast it in the pano frame to get a dense metric depth map (the range along each equirect
ray). Wall azimuths use the SAME convention as door_azimuth (the one that matches the pano
image, verified by the door-crop pipeline), so the depth aligns with the RGB pano.

Units: ZInD normalizes camera_height = 1; one layout unit = cam_h_m metres (= scale *
meters_per_coord). Floor is camera_height below the camera, ceiling (ceiling_height -
camera_height) above it.

    from sparsepano.datasets import zind_floor
    fl = zind_floor.ZindFloor(home/"zind_data.json", floor="floor_01")
    depth = render_layout_depth(fl, pano_stem, H=1024, W=2048)   # metres, (H,W)
"""
import numpy as np
from sparsepano.doors import door_dataset


def _room_polygon_local(fl, pano):
    """Room floor polygon in the pano's local horizontal frame (metres): (X=east, Z=fwd)."""
    info = fl.panos[pano]
    S = fl.meters_per_coord
    pos = np.asarray(info["pos"], float)
    vg = np.asarray(info["verts_global"], float)
    if vg.ndim != 2 or len(vg) < 3:
        return None
    xs, zs = [], []
    for v in vg:
        az = np.radians(door_dataset.door_azimuth(fl, pano, v))     # pano-frame azimuth (deg)
        r = float(np.linalg.norm(v - pos)) * S                      # horizontal distance (m)
        xs.append(r * np.sin(az)); zs.append(r * np.cos(az))
    return np.stack([xs, zs], 1)                                    # (N,2)


def _radial_wall_distance(poly, phis):
    """For each azimuth in `phis` (rad), nearest positive hit radius of the ray from the
    origin against the polygon edges. Vectorized over azimuths x edges."""
    D = np.stack([np.sin(phis), np.cos(phis)], 1)                   # (P,2) ray dirs
    A = poly                                                        # (N,2)
    B = np.roll(poly, -1, axis=0)                                   # (N,2) next vertex
    E = B - A                                                       # edge vectors (N,2)
    P = len(phis); N = len(poly)
    rho = np.full(P, np.inf)
    # solve O + t D = A + s E  for each (ray p, edge n):  [D, -E] [t s]^T = A
    Dx = D[:, 0][:, None]; Dy = D[:, 1][:, None]                    # (P,1)
    Ax = A[:, 0][None, :]; Ay = A[:, 1][None, :]                    # (1,N)
    Ex = E[:, 0][None, :]; Ey = E[:, 1][None, :]
    det = (-Dx * Ey) - (-Ex * Dy)                                  # (P,N) = Ex*Dy - Ey*Dx
    ok = np.abs(det) > 1e-9
    det_safe = np.where(ok, det, 1.0)
    t = ((-Ey) * Ax - (-Ex) * Ay) / det_safe                       # ray parameter (radius)
    s = (Dx * Ay - Dy * Ax) / det_safe                             # edge parameter
    valid = ok & (t > 1e-6) & (s >= -1e-6) & (s <= 1 + 1e-6)
    t = np.where(valid, t, np.inf)
    rho = t.min(axis=1)
    return rho                                                     # (P,)


def _az_column_mask(aza_deg, azb_deg, W, pad_deg):
    """Boolean (W,) of columns whose azimuth lies on the SHORT arc between the two bearings
    (padded), handling the +/-180 seam."""
    d = ((azb_deg - aza_deg + 180) % 360) - 180                   # signed shortest span
    lo = min(aza_deg, aza_deg + d) - pad_deg
    span = abs(d) + 2 * pad_deg
    au = (np.arange(W) / W - 0.5) * 360.0                         # azimuth per column (deg)
    return ((au - lo) % 360) <= span


def render_layout_depth(fl, pano, H=1024, W=2048, max_depth=15.0, mask_doors=True, pad_deg=4.0,
                        max_span_deg=110.0):
    """Dense equirect depth (metres, range along each ray) from the room layout.
    With mask_doors, the wall band at each door/opening azimuth is left EMPTY (depth 0) so we
    don't paste the neighbour room onto a solid wall — the passage stays a hole (to be filled by
    the neighbour's own points, or later by generation)."""
    info = fl.panos[pano]
    cam_h_m = info["cam_h_m"]
    h_floor = info["camera_height"] * cam_h_m                      # camera above floor (m)
    h_ceil = (info["ceiling_height"] - info["camera_height"]) * cam_h_m   # ceiling above camera
    poly = _room_polygon_local(fl, pano)
    if poly is None:
        return np.full((H, W), np.nan, np.float32)

    # cap by the room's own size: no valid surface is much farther than the room's diameter,
    # so anything beyond is a stray (camera annotated slightly outside its polygon).
    room_r = float(np.linalg.norm(poly, axis=1).max())
    max_depth = min(max_depth, 1.4 * room_r + h_ceil + h_floor)

    us = np.arange(W)
    phis = (us / W - 0.5) * 2 * np.pi                              # azimuth per column
    rho = _radial_wall_distance(poly, phis)                        # (W,) horizontal wall radius

    vs = np.arange(H)
    theta = (0.5 - vs / H) * np.pi                                 # elevation per row
    sin_t = np.sin(theta); cos_t = np.cos(theta)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_wall = rho[None, :] / np.clip(cos_t[:, None], 1e-6, None)   # (H,W) 3-D range to wall
        t_floor = np.where(sin_t < -1e-6, h_floor / np.clip(-sin_t, 1e-6, None), np.inf)
        t_ceil = np.where(sin_t > 1e-6, h_ceil / np.clip(sin_t, 1e-6, None), np.inf)
    t_vert = np.minimum(t_floor, t_ceil)[:, None] * np.ones((1, W))
    wall_hit = t_wall < t_vert                                     # (H,W) this ray hits a wall
    raw = np.minimum(t_wall, t_vert)
    # rays that miss the wall polygon (camera outside its annotated room for that azimuth) or
    # graze the horizon go to infinity -> DROP them (depth 0 = skipped) instead of flinging a
    # point out at max_depth, which is what littered the .ply.
    invalid = ~np.isfinite(raw) | (raw >= max_depth)
    depth = np.clip(raw, 0.05, max_depth).astype(np.float32)
    depth[invalid] = 0.0

    if mask_doors:
        S = fl.meters_per_coord; pos = np.asarray(info["pos"], float)
        segs = list(info.get("doors_global", [])) + list(info.get("openings_global", []))
        carve = np.zeros(W, bool)
        for p0, p1 in segs:
            aza = door_dataset.door_azimuth(fl, pano, p0)
            azb = door_dataset.door_azimuth(fl, pano, p1)
            span = abs(((azb - aza + 180) % 360) - 180) + 2 * pad_deg
            if span > max_span_deg:                              # degenerate/near-camera -> skip
                continue
            carve |= _az_column_mask(aza, azb, W, pad_deg)
        depth[wall_hit & carve[None, :]] = 0.0                    # empty the door/opening wall band
    return depth

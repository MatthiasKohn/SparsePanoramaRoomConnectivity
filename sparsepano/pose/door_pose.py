"""
Door-anchored relative pose.

A matched door is seen from both rooms. Two facts pin the pose without any blind
search:
  (1) the two cameras face the SAME wall from opposite sides, so for gravity-aligned
      panos the relative yaw is   theta ~= az_B - az_A + 180   (Ry maps dir(phi)->dir(phi+theta));
  (2) the door distance from each camera gives a translation init along the door
      bearing.
We then refine on the points that pass THROUGH the door (A's view of B's room),
which carry true parallax and forbid the camera-collapse (t=0) cheat.
"""
import numpy as np
from sparsepano.geometry import geom
from sparsepano.pose import pose as posemod


def _dir(az_deg):
    a = np.radians(az_deg)
    return np.array([np.sin(a), 0.0, np.cos(a)])


def _sector_wall(depth, az_deg, W, H, sector_deg=28, horizon_deg=22, min_depth=0.3):
    pts, _, _ = geom.backproject(depth, stride=2)
    r = np.linalg.norm(pts, axis=1)
    az = np.degrees(np.arctan2(pts[:, 0], pts[:, 2]))
    el = np.degrees(np.arcsin(np.clip(pts[:, 1] / r, -1, 1)))
    sect = (np.abs(((az - az_deg + 180) % 360) - 180) < sector_deg / 2)
    base = sect & (np.abs(el) < horizon_deg) & (r > min_depth)
    return (pts, r, az, el, sect), (np.percentile(r[base], 35) if base.sum() > 10 else 1.0)


def through_door_points(depth, az_deg, W, H, sector_deg=28, beyond=1.3,
                        min_depth=0.3, max_pts=1500, rng=None):
    (pts, r, az, el, sect), wall = _sector_wall(depth, az_deg, W, H, sector_deg, min_depth=min_depth)
    through = sect & (r > beyond * wall) & (r > min_depth)
    P = pts[through]
    if len(P) < 30:
        return None, wall
    if len(P) > max_pts:
        rng = rng or np.random.default_rng(0)
        P = P[rng.choice(len(P), max_pts, replace=False)]
    return P, wall


def recover(depth_a, depth_b, az_a_deg, az_b_deg, W, H, variant="V2",
            refine_yaw_deg=20.0, n_refine=9, tau=0.2, rng=None):
    P, d_a = through_door_points(depth_a, az_a_deg, W, H, rng=rng)
    if P is None:
        return None
    _, d_b = _sector_wall(depth_b, az_b_deg, W, H)
    yaw0 = np.radians(az_b_deg - az_a_deg + 180.0)
    cam_b_in_a = (d_a + d_b) * _dir(az_a_deg)             # where B sits, in A frame
    best = None
    for dy in np.radians(np.linspace(-refine_yaw_deg, refine_yaw_deg, n_refine)):
        R0 = geom.Ry(yaw0 + dy)
        Ti = np.eye(4); Ti[:3, :3] = R0; Ti[:3, 3] = -R0 @ cam_b_in_a
        o = posemod.recover(P, depth_b, Ti, W, H, variant=variant, max_nfev=120)
        res, _ = geom.residual(P, depth_b, o["R"], o["t"], o["s"], W, H)
        f = float((np.abs(res) < tau).mean()) if res.size else 0.0
        if best is None or f > best["inlier"]:
            best = {"R": o["R"], "t": o["t"], "s": o["s"], "inlier": f,
                    "n_through": len(P), "yaw_init_deg": float(np.degrees(yaw0))}
    return best

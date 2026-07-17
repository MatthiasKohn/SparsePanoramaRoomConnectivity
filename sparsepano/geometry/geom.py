"""
Equirectangular panorama geometry.

Conventions (ported verbatim from the validated Stage-1/2 code so results stay
comparable):
  - image (u,v): u in [0,W) longitude, v in [0,H) latitude (top=up)
  - ray = [cos(th)sin(phi), sin(th), cos(th)cos(phi)]  with
        phi   = (u/W - 0.5) * 2pi
        theta = (0.5 - v/H) * pi
  - Y is the vertical (up) axis. For gravity-aligned panoramas the relative
    rotation between two cameras is therefore a pure rotation about Y (yaw).
  - A point p_a in camera-A frame maps to camera-B frame as:  p_b = s * R @ p_a + t
"""
import numpy as np


def pixel_to_ray(u, v, W, H):
    phi = (u / W - 0.5) * 2 * np.pi
    theta = (0.5 - v / H) * np.pi
    ct = np.cos(theta)
    return np.stack([ct * np.sin(phi), np.sin(theta), ct * np.cos(phi)], -1)


def backproject(depth, stride=1):
    """Equirect depth -> 3D points in the camera frame (+ their pixel coords)."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    us = us.ravel().astype(np.float32)
    vs = vs.ravel().astype(np.float32)
    d = depth[vs.astype(int), us.astype(int)]
    ok = np.isfinite(d) & (d > 0.1)
    us, vs, d = us[ok], vs[ok], d[ok]
    rays = pixel_to_ray(us, vs, W, H)
    return rays * d[:, None], us, vs


def Ry(alpha):
    """Yaw rotation about the vertical (Y) axis."""
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def so3_log_angle(R):
    """Rotation angle (deg) of R."""
    c = (np.trace(R) - 1) / 2
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def project_to_pano(points, W, H):
    """3D points -> (u, v, range) on an equirectangular image."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    r = np.clip(r, 1e-6, None)
    phi = np.arctan2(x, z)
    theta = np.arcsin(np.clip(y / r, -1, 1))
    u = (phi / (2 * np.pi) + 0.5) * W
    v = (0.5 - theta / np.pi) * H
    return u, v, r


def sample_bilinear(img, u, v):
    """Bilinear sample with border padding. NaN where source depth invalid."""
    H, W = img.shape
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    u1 = u0 + 1; v1 = v0 + 1
    au = u - u0; av = v - v0
    def g(vv, uu):
        return img[vv, uu]
    val = (g(v0, u0) * (1 - au) * (1 - av) + g(v0, u1) * au * (1 - av)
           + g(v1, u0) * (1 - au) * av + g(v1, u1) * au * av)
    return val


def residual(points_a, depth_b, R, t, s, W, H):
    """Per-point depth-consistency residual (range_in_B - measured_depth_B)."""
    pb = s * (points_a @ R.T) + t
    u, v, r = project_to_pano(pb, W, H)
    meas = sample_bilinear(depth_b, u, v)
    valid = np.isfinite(meas) & (meas > 0.1)
    return (r - meas)[valid], valid


# ---- relative pose & error metrics --------------------------------------
def rel_pose(pa, pb):
    """T_gt mapping A-frame -> B-frame. pa/pb are dicts with 'R','t'."""
    R = pb["R"] @ pa["R"].T
    t = pb["t"] - R @ pa["t"]
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def pose_error(R, t, T_gt):
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
    rot = so3_log_angle(R.T @ R_gt)
    return rot, float(np.linalg.norm(t - t_gt))


def covisible_columns(depth_a, depth_b, T_gt, W, H, stride=6, tau=0.25,
                      min_depth=0.3, min_count=3):
    """Boolean[W]: azimuth columns of A that contain >= min_count pixels which,
    transformed to B at GT pose, land consistently on B's surface (the true
    'see-through toward B' region). Used as detection ground truth."""
    pts, us, vs = backproject(depth_a, stride=stride)
    if len(pts) == 0:
        return np.zeros(W, bool), None
    R, t = T_gt[:3, :3], T_gt[:3, 3]
    pb = pts @ R.T + t
    u, v, r = project_to_pano(pb, W, H)
    meas = sample_bilinear(depth_b, u, v)
    a_range = np.linalg.norm(pts, axis=1)
    ok = (np.isfinite(meas) & (meas > 0.1) & (a_range > min_depth)
          & (np.abs(r - meas) < tau))
    cols = np.zeros(W, int)
    uu = np.clip(us[ok].astype(int), 0, W - 1)
    np.add.at(cols, uu, 1)
    return cols >= min_count, pts[ok]


def decompose_gravity(R):
    """How 'yaw-like' is R? Returns (total_angle_deg, off_yaw_deg).

    off_yaw = angle by which R tilts the vertical axis (0 => pure yaw / gravity
    aligned). Large off_yaw would invalidate the yaw-only assumption.
    """
    ey = np.array([0.0, 1.0, 0.0])
    tilt = np.rad2deg(np.arccos(np.clip(np.dot(R @ ey, ey), -1, 1)))
    return so3_log_angle(R), tilt

"""
Equirectangular <-> perspective sampling.

Most off-the-shelf door detectors / segmenters are trained on normal photos, not
360 imagery. So we cut a panorama into a few overlapping perspective views, run the
detector on those, and map detections back to a panorama azimuth. Convention
matches geom.py (phi = atan2(x, z), Y up).
"""
import numpy as np
import cv2


def e2p(equi, yaw_deg, pitch_deg=0.0, fov_deg=90.0, out_hw=(720, 720)):
    """Sample a perspective view (camera looks toward +Z rotated by yaw/pitch)."""
    H, W = equi.shape[:2]
    Ho, Wo = out_hw
    f = 0.5 * Wo / np.tan(np.radians(fov_deg) / 2)
    xs, ys = np.meshgrid(np.arange(Wo), np.arange(Ho))
    x = xs - Wo / 2.0
    y = -(ys - Ho / 2.0)                 # image y down -> world y up
    z = np.full_like(x, f, dtype=float)
    vec = np.stack([x, y, z], -1).astype(float)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    vec = vec @ (Ry @ Rx).T
    lon = np.arctan2(vec[..., 0], vec[..., 2])
    lat = np.arcsin(np.clip(vec[..., 1], -1, 1))
    u = ((lon / (2 * np.pi)) + 0.5) * W
    v = (0.5 - lat / np.pi) * H
    return cv2.remap(equi, u.astype(np.float32), v.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def crop_x_to_azimuth(x_in_crop, yaw_deg, fov_deg, crop_w):
    """Map a detection's horizontal centre (px in the crop) back to pano azimuth (deg)."""
    f = 0.5 * crop_w / np.tan(np.radians(fov_deg) / 2)
    ang = np.degrees(np.arctan2((x_in_crop - crop_w / 2.0), f))
    return ((yaw_deg + ang + 180) % 360) - 180


def ring_views(equi, n=8, fov_deg=90.0, out_hw=(720, 720), pitch_deg=0.0):
    """n overlapping perspective views around the horizon. Returns list of
    (yaw_deg, image)."""
    return [(yaw, e2p(equi, yaw, pitch_deg, fov_deg, out_hw))
            for yaw in np.linspace(0, 360, n, endpoint=False)]

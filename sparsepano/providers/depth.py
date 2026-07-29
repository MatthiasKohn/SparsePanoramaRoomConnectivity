"""Depth provider — the per-room geometry building block.

get_depth(fl, home, pano, H, W, model, ...) -> equirect metric depth (H,W) in the pano frame.

Models:
  gt_layout : GT room-layout depth only (exact walls/floor/ceiling, no furniture). The pure oracle.
  fused     : GT layout + PaGeR fused (furniture where it sticks out in front of walls). Best-looking.
  pager     : PaGeR monocular depth only.
  dap       : DAP monocular depth only.
Missing external depth falls back to the layout depth (with a note upstream via the return being layout).
"""
from pathlib import Path
import numpy as np
import cv2

from sparsepano.geometry import layout_depth


def _load_ext(home, pano, H, W, sub):
    p = Path(home) / sub / f"{pano}.npy"
    if not p.exists():
        return None
    d = np.squeeze(np.load(p).astype(np.float32))
    if d.ndim == 3:
        d = d[..., 0] if d.shape[-1] == 1 else d[0]
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    return d


def fuse_layout_pager(layout, ext, margin=0.15, max_front=3.0):
    """Clean layout walls + `ext` (PaGeR/DAP) where it reads meaningfully IN FRONT of the wall
    (furniture). `ext` is robustly scale/shift-aligned to the layout; capped by max_front so a
    misread recess (stairwell/opening) can't fling a blob into the room. Doorway holes stay holes."""
    v = (layout > 0.1) & (ext > 0.1) & np.isfinite(ext)
    if v.sum() < 200:
        return layout
    x, y = ext[v], layout[v]; a, b = 1.0, 0.0
    for _ in range(3):
        r = np.abs(a * x + b - y); w = 1.0 / (r + 0.1)
        A = np.stack([x * w, w], 1)
        sol, *_ = np.linalg.lstsq(A, y * w, rcond=None); a, b = float(sol[0]), float(sol[1])
    pa = a * ext + b
    fused = layout.copy()
    obj = v & (pa < layout - margin) & (pa > layout - max_front) & (pa > 0.1)
    fused[obj] = pa[obj]
    return fused


def get_depth(fl, home, pano, H, W, model="fused", max_depth=15.0, mask_doors=True,
              carve_doors=False, pager_sub="pager_depth/depth_meters", dap_sub="dap_depth/depth_meters"):
    layout = layout_depth.render_layout_depth(fl, pano, H, W, max_depth=max_depth,
                                              mask_doors=mask_doors, carve_doors=carve_doors)
    if model in ("gt", "gt_layout", "layout"):
        return layout
    if model in ("fused", "gt_pager"):
        pg = _load_ext(home, pano, H, W, pager_sub)
        return fuse_layout_pager(layout, pg) if pg is not None else layout
    if model == "pager":
        pg = _load_ext(home, pano, H, W, pager_sub)
        return pg if pg is not None else layout
    if model == "dap":
        d = _load_ext(home, pano, H, W, dap_sub)
        return d if d is not None else layout
    raise ValueError(f"unknown depth_model {model!r}")

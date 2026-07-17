"""Layout metrics."""

from __future__ import annotations

import numpy as np


def position_error(pred_xy: np.ndarray, gt_xy: np.ndarray) -> dict[str, float]:
    pred_xy = np.asarray(pred_xy, dtype=float)
    gt_xy = np.asarray(gt_xy, dtype=float)
    if len(pred_xy) == 0:
        return {"mean_m": float("nan"), "median_m": float("nan")}
    err = np.linalg.norm(pred_xy - gt_xy, axis=1)
    return {"mean_m": float(np.mean(err)), "median_m": float(np.median(err))}


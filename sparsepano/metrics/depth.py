"""Depth/geometry metrics."""

from __future__ import annotations

import numpy as np


def depth_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    if valid is None:
        valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    pred_v = pred[valid].astype(float)
    gt_v = gt[valid].astype(float)
    if pred_v.size == 0:
        return {"absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}
    ratio = np.maximum(pred_v / np.maximum(gt_v, 1e-9), gt_v / np.maximum(pred_v, 1e-9))
    return {
        "absrel": float(np.mean(np.abs(pred_v - gt_v) / np.maximum(gt_v, 1e-9))),
        "rmse": float(np.sqrt(np.mean((pred_v - gt_v) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


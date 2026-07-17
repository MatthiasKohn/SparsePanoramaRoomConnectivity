"""Pose and flip metrics."""

from __future__ import annotations

import numpy as np


def rotation_error_deg(r_pred: np.ndarray, r_gt: np.ndarray) -> float:
    rel = r_pred[:3, :3] @ r_gt[:3, :3].T
    cos = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def flip_accuracy(pred: list[int] | np.ndarray, gt: list[int] | np.ndarray) -> float:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.size == 0:
        return float("nan")
    return float(np.mean(pred == gt))


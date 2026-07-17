"""Connectivity metrics."""

from __future__ import annotations

import numpy as np


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0 or y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score)
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y_true.sum()), 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def operating_point(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "threshold": float("nan")}
    order = np.argsort(-y_score)
    yt = y_true[order]
    scores = y_score[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y_true.sum()), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    i = int(np.argmax(f1))
    return {
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i]),
        "threshold": float(scores[i]),
    }


def edge_diagnostics(
    gt_edges: set[tuple[str, str]],
    pred_edges: set[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    norm_gt = {tuple(sorted(e)) for e in gt_edges}
    norm_pred = {tuple(sorted(e)) for e in pred_edges}
    return {
        "matched": sorted(norm_gt & norm_pred),
        "missing": sorted(norm_gt - norm_pred),
        "wrong": sorted(norm_pred - norm_gt),
    }


"""Door detection metrics."""

from __future__ import annotations

from dataclasses import dataclass


def circular_diff_deg(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


@dataclass(frozen=True)
class DoorDetectionDiagnostics:
    matched: list[tuple[str, float, float]]
    missed: list[tuple[str, float]]
    false_positives: list[tuple[str, float]]
    precision: float
    recall: float


def match_door_bearings(
    gt_by_pano: dict[str, list[float]],
    pred_by_pano: dict[str, list[float]],
    tol_deg: float,
) -> DoorDetectionDiagnostics:
    matched: list[tuple[str, float, float]] = []
    missed: list[tuple[str, float]] = []
    false_pos: list[tuple[str, float]] = []

    for pano_id in sorted(set(gt_by_pano) | set(pred_by_pano)):
        gt = list(gt_by_pano.get(pano_id, []))
        pred = list(pred_by_pano.get(pano_id, []))
        used: set[int] = set()
        for p in pred:
            best = None
            best_err = tol_deg
            for i, g in enumerate(gt):
                if i in used:
                    continue
                err = circular_diff_deg(p, g)
                if err <= best_err:
                    best = i
                    best_err = err
            if best is None:
                false_pos.append((pano_id, p))
            else:
                used.add(best)
                matched.append((pano_id, gt[best], p))
        for i, g in enumerate(gt):
            if i not in used:
                missed.append((pano_id, g))

    precision = len(matched) / max(len(matched) + len(false_pos), 1)
    recall = len(matched) / max(len(matched) + len(missed), 1)
    return DoorDetectionDiagnostics(matched, missed, false_pos, precision, recall)


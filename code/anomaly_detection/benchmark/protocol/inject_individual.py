from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Any, Sequence

from .schemas import AnomalyLabel, build_sample_id

DEFAULT_INDIVIDUAL_ANOMALIES = (
    "route_shift",
    "speed_spike",
    "stop_or_slowdown",
    "jump",
    "shape_warp",
    "modal_offset",
)
MODALITIES = ("fused", "rgb", "thermal")


def inject_individual_anomalies(
    trajectories: list[dict[str, Any]],
    anomaly_fraction: float,
    seed: int,
    anomaly_types: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[AnomalyLabel]]:
    if not 0 <= anomaly_fraction <= 1:
        raise ValueError("anomaly_fraction must be between 0 and 1")

    selected_types = tuple(anomaly_types or DEFAULT_INDIVIDUAL_ANOMALIES)
    _validate_types(selected_types, DEFAULT_INDIVIDUAL_ANOMALIES)

    injected = deepcopy(trajectories)
    candidates = [index for index, item in enumerate(injected) if item.get("points")]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    target_count = _selection_count(len(candidates), anomaly_fraction)

    labels: list[AnomalyLabel] = []
    type_cursor = 0
    for index in candidates:
        if len(labels) >= target_count:
            break
        trajectory = injected[index]
        for offset in range(len(selected_types)):
            anomaly_type = selected_types[(type_cursor + offset) % len(selected_types)]
            if _apply_individual_anomaly(trajectory, anomaly_type, rng):
                labels.append(_build_label(trajectory, anomaly_type, seed))
                type_cursor = (type_cursor + offset + 1) % len(selected_types)
                break

    return injected, labels


def _validate_types(types: Sequence[str], allowed: Sequence[str]) -> None:
    unknown = sorted(set(types) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported anomaly_types: {unknown}")


def _selection_count(candidate_count: int, fraction: float) -> int:
    if candidate_count == 0 or fraction <= 0:
        return 0
    return min(candidate_count, max(1, math.ceil(candidate_count * fraction)))


def _apply_individual_anomaly(
    trajectory: dict[str, Any], anomaly_type: str, rng: random.Random
) -> bool:
    points = trajectory["points"]
    if anomaly_type == "route_shift":
        changed = _shift_points(points, dx=_signed(rng, 6.0), dy=_signed(rng, 4.0))
    elif anomaly_type == "speed_spike":
        changed = _speed_spike(points)
    elif anomaly_type == "stop_or_slowdown":
        changed = _stop_or_slowdown(points)
    elif anomaly_type == "jump":
        midpoint = len(points) // 2
        changed = _shift_points(
            [points[midpoint]], dx=_signed(rng, 10.0), dy=_signed(rng, 8.0)
        )
    elif anomaly_type == "shape_warp":
        changed = _shape_warp(points)
    elif anomaly_type == "modal_offset":
        changed = _modal_offset(points, dx=_signed(rng, 3.0), dy=_signed(rng, 2.0))
    else:
        changed = False
    if changed:
        trajectory.setdefault("metadata", {})["anomaly_type"] = anomaly_type
    return changed


def _shift_points(points: Sequence[dict[str, Any]], dx: float, dy: float) -> bool:
    changed = False
    for point in points:
        for center in _centers(point):
            center[0] += dx
            center[1] += dy
            changed = True
    return changed


def _speed_spike(points: list[dict[str, Any]]) -> bool:
    if len(points) < 2:
        return False
    anchor = _primary_center(points[0])
    if anchor is None:
        return False
    changed = False
    for point in points[1:]:
        target = _primary_center(point)
        if target is None:
            continue
        dx = target[0] - anchor[0]
        dy = target[1] - anchor[1]
        for center in _centers(point):
            center[0] += dx
            center[1] += dy
            changed = True
    return changed


def _stop_or_slowdown(points: list[dict[str, Any]]) -> bool:
    if len(points) < 2:
        return False
    anchor_index = max(0, len(points) // 2 - 1)
    anchor = _primary_center(points[anchor_index])
    if anchor is None:
        return False
    changed = False
    for point in points[anchor_index + 1 :]:
        for center in _centers(point):
            if center[0] != anchor[0] or center[1] != anchor[1]:
                center[0] = anchor[0]
                center[1] = anchor[1]
                changed = True
    return changed


def _shape_warp(points: list[dict[str, Any]]) -> bool:
    centers = [_primary_center(point) for point in points]
    valid = [center for center in centers if center is not None]
    if not valid:
        return False
    mean_x = sum(center[0] for center in valid) / len(valid)
    mean_y = sum(center[1] for center in valid) / len(valid)
    changed = False
    for point in points:
        for center in _centers(point):
            new_x = mean_x + (center[0] - mean_x) * 1.25
            new_y = mean_y + (center[1] - mean_y) * 0.5
            if center[0] != new_x or center[1] != new_y:
                center[0] = new_x
                center[1] = new_y
                changed = True
    return changed


def _modal_offset(points: list[dict[str, Any]], dx: float, dy: float) -> bool:
    changed = False
    for point in points:
        for modality in ("rgb", "thermal"):
            center = _center(point.get(modality))
            if center is not None:
                center[0] += dx
                center[1] += dy
                changed = True
    return changed


def _centers(item: dict[str, Any]) -> list[list[float]]:
    return [center for modality in MODALITIES if (center := _center(item.get(modality))) is not None]


def _center(modality_value: Any) -> list[float] | None:
    if isinstance(modality_value, dict):
        center = modality_value.get("center_xy")
        if isinstance(center, list) and len(center) >= 2:
            return center
    return None


def _primary_center(point: dict[str, Any]) -> list[float] | None:
    for modality in MODALITIES:
        center = _center(point.get(modality))
        if center is not None:
            return center
    return None


def _signed(rng: random.Random, magnitude: float) -> float:
    return magnitude if rng.random() >= 0.5 else -magnitude


def _build_label(
    trajectory: dict[str, Any], anomaly_type: str, seed: int
) -> AnomalyLabel:
    points = trajectory.get("points", [])
    frames = [int(point["frame_id"]) for point in points if "frame_id" in point]
    sequence = str(trajectory.get("sequence", ""))
    track_id = str(trajectory.get("track_id", ""))
    sample_id = str(trajectory.get("sample_id") or build_sample_id(sequence, track_id))
    return AnomalyLabel(
        sample_id=sample_id,
        sequence=sequence,
        track_id=track_id,
        frame_start=min(frames) if frames else 0,
        frame_end=max(frames) if frames else 0,
        label=1,
        anomaly_type=anomaly_type,
        injection_seed=seed,
        metadata={"source": "individual_injection"},
    )

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd


PREFERRED_MODALITIES = ("fused", "rgb", "thermal")
IDENTIFIER_COLUMNS = ("sample_id", "sequence", "track_id")
FEATURE_COLUMNS = (
    "duration_frames",
    "num_points",
    "path_length",
    "displacement",
    "mean_speed",
    "max_speed",
    "std_speed",
    "mean_acceleration",
    "max_acceleration",
    "mean_turn_angle",
    "max_turn_angle",
    "bbox_area_mean",
    "bbox_area_std",
    "modal_offset_mean",
    "modal_offset_max",
)
OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS


def extract_center_sequence(
    trajectory: dict,
    preferred_modalities: tuple[str, ...] = PREFERRED_MODALITIES,
) -> list[tuple[int, float, float]]:
    sequence: list[tuple[int, float, float]] = []
    for point in trajectory.get("points", []):
        center = None
        for modality in preferred_modalities:
            center = _center_from_state(point.get(modality))
            if center is not None:
                break
        if center is None or "frame_id" not in point:
            continue
        sequence.append((int(point["frame_id"]), center[0], center[1]))
    return sorted(sequence, key=lambda item: item[0])


def build_handcrafted_feature_row(trajectory: dict) -> dict[str, Any]:
    sequence = extract_center_sequence(trajectory)
    frames = [frame_id for frame_id, _, _ in sequence]
    centers = [(x, y) for _, x, y in sequence]
    step_distances = _step_distances(sequence)
    speeds = _step_speeds(sequence)
    accelerations = _accelerations(speeds)
    turn_angles = _turn_angles(centers)
    bbox_areas = _bbox_areas(trajectory)
    modal_offsets = _modal_offsets(trajectory)

    row: dict[str, Any] = {
        "sample_id": _sample_id(trajectory),
        "sequence": str(trajectory.get("sequence", "")),
        "track_id": str(trajectory.get("track_id", "")),
        "duration_frames": int(frames[-1] - frames[0] + 1) if frames else 0,
        "num_points": int(len(sequence)),
        "path_length": _sum(step_distances),
        "displacement": _distance(centers[0], centers[-1]) if len(centers) >= 2 else 0.0,
        "mean_speed": _mean(speeds),
        "max_speed": _max(speeds),
        "std_speed": _std(speeds),
        "mean_acceleration": _mean(accelerations),
        "max_acceleration": _max(accelerations),
        "mean_turn_angle": _mean(turn_angles),
        "max_turn_angle": _max(turn_angles),
        "bbox_area_mean": _mean(bbox_areas),
        "bbox_area_std": _std(bbox_areas),
        "modal_offset_mean": _mean(modal_offsets),
        "modal_offset_max": _max(modal_offsets),
    }
    return {key: _finite_default(value) for key, value in row.items()}


def build_handcrafted_feature_table(trajectories: Iterable[dict]) -> pd.DataFrame:
    rows = [build_handcrafted_feature_row(trajectory) for trajectory in trajectories]
    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if table.empty:
        return table
    for column in FEATURE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    return table.sort_values("sample_id", kind="stable").reset_index(drop=True)


def _sample_id(trajectory: dict) -> str:
    sample_id = trajectory.get("sample_id")
    if sample_id not in (None, ""):
        return str(sample_id)
    sequence = str(trajectory.get("sequence", ""))
    track_id = str(trajectory.get("track_id", ""))
    return f"{sequence}:{track_id}" if sequence or track_id else ""


def _center_from_state(state: Any) -> tuple[float, float] | None:
    if not isinstance(state, dict):
        return None
    center = state.get("center_xy")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return None
    try:
        x = float(center[0])
        y = float(center[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _step_distances(sequence: list[tuple[int, float, float]]) -> list[float]:
    return [
        _distance((x0, y0), (x1, y1))
        for (_, x0, y0), (_, x1, y1) in zip(sequence, sequence[1:])
    ]


def _step_speeds(sequence: list[tuple[int, float, float]]) -> list[float]:
    speeds: list[float] = []
    for (frame0, x0, y0), (frame1, x1, y1) in zip(sequence, sequence[1:]):
        delta_frames = max(frame1 - frame0, 1)
        speeds.append(_distance((x0, y0), (x1, y1)) / float(delta_frames))
    return speeds


def _accelerations(speeds: list[float]) -> list[float]:
    return [abs(next_speed - speed) for speed, next_speed in zip(speeds, speeds[1:])]


def _turn_angles(centers: list[tuple[float, float]]) -> list[float]:
    angles: list[float] = []
    for start, middle, end in zip(centers, centers[1:], centers[2:]):
        vector_a = (middle[0] - start[0], middle[1] - start[1])
        vector_b = (end[0] - middle[0], end[1] - middle[1])
        norm_a = math.hypot(*vector_a)
        norm_b = math.hypot(*vector_b)
        if norm_a == 0.0 or norm_b == 0.0:
            continue
        cosine = (
            (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1])
            / (norm_a * norm_b)
        )
        angles.append(math.acos(max(-1.0, min(1.0, cosine))))
    return angles


def _bbox_areas(trajectory: dict) -> list[float]:
    areas: list[float] = []
    for point in trajectory.get("points", []):
        for modality in PREFERRED_MODALITIES:
            state = point.get(modality)
            area = _bbox_area(state)
            if area is not None:
                areas.append(area)
                break
    return areas


def _bbox_area(state: Any) -> float | None:
    if not isinstance(state, dict):
        return None
    box = state.get("bbox_xyxy", state.get("bbox"))
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in box[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    width = x1 - x0
    height = y1 - y0
    if width < 0.0 or height < 0.0:
        width = x1
        height = y1
    return max(width, 0.0) * max(height, 0.0)


def _modal_offsets(trajectory: dict) -> list[float]:
    offsets: list[float] = []
    for point in trajectory.get("points", []):
        centers = [
            center
            for center in (_center_from_state(point.get(modality)) for modality in PREFERRED_MODALITIES)
            if center is not None
        ]
        offsets.extend(_distance(first, second) for first, second in combinations(centers, 2))
    return offsets


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(math.hypot(second[0] - first[0], second[1] - first[1]))


def _sum(values: list[float]) -> float:
    return float(np.sum(values)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _finite_default(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value
